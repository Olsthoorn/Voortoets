# %% [markdown]

# The idea is to develop a small finite difference model that is virtually as easily applicable and fast in practice as the use of analytical formulas.

# * the model will be phreatic (1 hydraulic layer, perhaps up to 3 model layers to include partial penetration of wells)
# * layer data will be extracted from a geological database.
# * surface water data will be extracted from a shapefile
# * the model will just compute drawdowns and water budget.
# * the effect of the boundaries of the model will be verified.
#
# The model will be a class that is intialized using the following parameters: xc=None, yc=None, R=7500, dx=5, dxmult
#
# * The extracted layer data will be converted to Depth, kx, ky and Sy
# * Surfface water shapefile will be converted to line segments with fields x1,y1, x2,y2, w, c
#
# The user specifies his/her intervention using
#
# * x, y, location (z1) (z2) depth in m, Q[...Qn] (monthly values)
# more than one well is possible
#
# Alternatively:
#
# * ((x1, y1), (x2, y2) etc as a contour or a polyline.
# * combined with the desired total extraction
# * total drawdown (areal drawdown or drawdown along section)
#
# Results:
#
# * Drawdown contours (after each month and at final stage (or steady state))
# * Water budget: extraction, extraction from surface water, extraction from model boundary, extraction from storage
#
# * contours plotted on map (mat should contain contours of vulnerable areas)
#
# * Contours are also compare with those from analytical formula (Hantush).


# %%
# --- standard library
import numpy as np
import matplotlib.pyplot as plt
import etc

# --- Local / project
from fdm.src.mfgrid import Grid
from fdm.src.fdm3t import Fdm3t, fdm3t
from fdm.src.fdm3t import dtypeQ, dtypeH, dtypeGHB

from select_surf_wat_15x15km import get_water_length_per_cell, clip_water_15km, water_length_per_cell


# %%

def get_grid(ctr, wc=5, w=15000., N=100, z=np.array([0, -30.])):
    """Return grid object containing the rectangular model grid.
    
    Parameters
    ----------
    ctr: tuple x,y in crs: 30310
        Center of the grid in crs 30370 (Belgium)
    wc: float
        minimum cell width
    w: float
       with of model (xMax - xMin) = (yMax - yMin)
    N: int
        nx, ny of model grid
    z: np.ndarray
        elevation of water table aquifer
    """
    x0, y0 = ctr

    # --- increasing coordinates from center
    xi = np.logspace(np.log10(wc / 2), np.log10(w / 2), int(N // 2))
    xi = np.hstack((-xi[::-1], xi))
    
    # --- real world coordinates of vertical grid lines
    x = x0 + xi
    
    # --- real world coordinaters of horizontal grid lines
    y = y0 - xi
    
    # --- create grid object
    gr = Grid(x, y, z, axial=False)
    return gr


class Vt_model():
    """Voortoets model, simple version, 1 layer FDM."""

    def __init__(self, ctr, wc=5, w=15000, z=np.array([0, -30.]), N=100, Q=None):
        """
        Parameters
        ----------
        ctr: (float, float)
            coordinates of model center (center crs=30370 (Belgium)
        wc: float
            minimum cell width and height
        w: float
            model width and height (xmax - xmin = ymax - ymin)
        z: np.ndarray
            layer elevations
        N: int
            nx and ny (cells)
        """
        self.gr = get_grid(ctr=ctr, wc=wc, w=w, N=N, z=z)
        self.set_screen(top=None, bot=None)
        self.set_IDOMAIN()
        self.set_model_params()        
        self.set_initial_conditions()
        self.set_wel(Q)
        self.set_ghb()

        
    def set_screen(self, top=None, bot=None):
        """Set elevation of well screen.
        
        Parameters
        ----------
        top: float | None
            top of well screen [m relative to groound sruface i.e. [<=0]
            Default = None --> ground surface.
        bot: float | None 
            bot of well screen [m relative to ground surface [<0]]
            Default = None -->  base of phreatic aquifer.
        """
        # --- make sure screen_top is above screen_bot
        gr = self.gr
        top = gr.z[ 0] if top is None else top
        bot = gr.z[-1] if bot is None else bot
        if top < bot:
            top, bot = bot, top            
        if top > gr.z[0]:
            raise ValueError(f"Screen top must be <= {gr.z[0]} or None")
        if bot < gr.z[-1]:
            raise ValueError(f"Screen bot must be >= {gr.z[-1]} or None")
                    
        # --- if relevant, add screen_top and or screen_bot to layer elevations
        z = list(gr.z)
        
        if top < gr.z[0] - 5:
            z = sorted(z.append(top))
        if bot > gr.z[1] + 5:
            z = sorted(z.append(bot))
            
        self.scr_top = top
        self.scr_bot = bot
        
        # --- finally regenerate the grid object using the new z
        self.gr = Grid(gr.x, gr.y, np.array(z)[::-1], axial=False)
        return None
    
    def set_model_params(self):
        """Set model parameters (not the boundary conditions).
        
        These parameters should come from a database that yield them
        when given the coordinates of a point in Belgium in crs 31370
        
        """
        kx, ky, kz, Ss, Sy = 10., 10., 2., 0.0001, 0.2
        
        gr = self.gr
        
        # --- Generate the model parameters (for the time being)
        self.kx =gr.const(kx)
        self.ky = gr.const(ky)
        self.kz = gr.const(kz)
        self.Ss = gr.const(Ss)
        self.Ss[0] = gr.const(Sy / gr.dz[0])        
        return None
    
    def set_IDOMAIN(self):
        self.IDOMAIN = self.gr.const(1, dtype=int)
        
    def set_initial_conditions(self):
        self.HI = self.gr.const(0.)

    def set_wel(self, Q):
        """Set wells (FQ).
        
        Parameters
        ----------
        Q: iterative
            Up to 6 monthly Q values. The extraction will be at xy_ctr
            extraction is negative, infiltration is postive
                       
        """
        gr = self.gr
        Q = np.atleast_1d(Q)        
        zm = 0.5 * (self.scr_top + self.scr_bot)
        xyz = np.array([[gr.x.mean(), gr.y.mean(), zm]])
        Idx = gr.Iglob_from_xyz(xyz)
        self.WEL = {}
        for i, Q_ in enumerate(Q):
            fq = np.zeros(len(Idx), dtype=dtypeQ)            
            fq['I'] = Idx
            fq['q'] = Q_
            self.WEL[i] = fq
        return None
               
    def set_chd(self, h):
        """Set CHD (fixed heads).
        
        Parameters
        ----------
        h : iterable of monthly heads
        """
        gr = self.gr
        self.CHD = {}
        h = np.atleast_1d(h)
        zm = 0.5 * (self.screen_top + self.screen_bot)
        xyz = np.array([[gr.x.mean(), gr.y.mean(), zm]])
        Idx = gr.Iglob_from_xyz(xyz)
        for i, h_ in enumerate(h):
            fh = np.zeros(len(Idx), dtype=dtypeH)            
            fh['I'] = Idx
            fh['h'] = h_
            self.CHD[i] = fh
        return None
    
    def set_ghb(self, w=5, c=5):
        """Set general head boundary conditions.
        
        Parameters
        ----------
        w: float
            with the all surface waters [m]
        c: float
            bottom resistance of all surface waters [d]
        """        
        gr = self.gr
        
        # --- get surface water length per model cell.
        self.clipped, tile_gdf = clip_water_15km(gr, target_crs="EPSG:31370")
        L = water_length_per_cell(gr, self.clipped, tile_gdf)['water_length_m']
                        
        ghb = np.zeros(gr.nod, dtype=dtypeGHB)
        ghb['I'] = gr.NOD.flatten()
        ghb['C'] = L * w / c
        ghb['h'] = 0.
        
        ghb = ghb[ghb['C'] > 0]
        self.GHB = {0: ghb}
        return None

        
    def run_mdl(self, t=np.logspace(-3, np.log10(180))):
        self.CHD = None       
        self.out = fdm3t(gr=self.gr, t=t, k=(self.kx, self.ky, self.kz),
                    c=None, ss=self.Ss, fh=self.CHD, ghb=self.GHB,
                    fq=self.WEL, hi=self.HI, idomain=self.IDOMAIN)
        return None
        
    def evaluate(self, t=None, ax=None, levels=None):
        """Plot and return the model results for the user.
        
        Parameters
        ----------
        t: float
            at time(s) for which contours are desired.
        ax: Axes object or None
            axes used for plot. Of None a new axes is generated.
        levels: iterable
            the levels to show on the contour plot [m, drawdown]        
        """
        if levels is None:
            levels=-np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
        levels.sort()
            
        
        if t is None:
            t = self.out['t'][-1]
        
        it = np.where(self.out['t'] <= t)[0][-1]
        if ax is None:
            title = f"Verlaging [m] at t={self.out['t'][it]:.0f} d after start extraction."
            ax = etc.newfig(title, 'x [m]', 'y [m]')
        
        self.clipped.plot(ax=ax, color='blue', linewidth=1)
        
        phi = self.out['Phi'][it][0]
        gr = self.gr
        Cs = ax.contour(gr.xm, gr.ym, phi, colors='k', levels=levels)
        ax.clabel(Cs, inline=1, fontsize=10, fmt='%1.2f')
        
        # --- water budget: all in the out dictionary
        # self.out.Q
        
        # --- The extractions
        print("Qwel = {self.out['Q'].sum():.0f} m3/d")
        
        # --- Infiltration from surface water
        # self.out['Qriv']
        
        # --- Flow over the model boundary
        print(f"Totale Qghb = {self.out['Qghb'].sum():.0f} m3/d")
        
        # --- Storage
        print(f"Qstorage = {self.out['Qs'].sum():.0f} m3/d")
        
        return None

# %%
ctr = (193919, 194774)

vtmdl = Vt_model(ctr, wc=5, w=15000, z=np.array([0, -30.]), N=100, Q=-2400.)

vtmdl.run_mdl(t=np.logspace(-3, np.log10(180)))
vtmdl.evaluate(t=None, ax=None, levels=None)
plt.show()
print('Done!')
