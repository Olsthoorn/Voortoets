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
import os
import numpy as np
import matplotlib.pyplot as plt
import etc

# --- Local / project
from fdm.src.mfgrid import Grid
from fdm.src.fdm3t import fdm3t
from fdm.src.fdm3t import dtypeQ, dtypeH, dtypeGHB
from src.vtl_layering import get_layering
from select_surf_wat_15x15km import clip_water_15km, water_length_per_cell
from shapely.geometry import Point


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

def grid_fr_wells(wells, L=15000., N=50):
    """Return grid from wells as specified in userinput.json."""
    
    coords = []
    for well in wells:
        coords.append((well['geometry'].x, well['geometry'].y))
    coords = np.array(coords)
    xC, yC = np.array(coords).mean(axis=0)
    
    # --- select coordinates such that wells fall inside model cells
    rm = np.logspace(np.log10(10), np.log10(L / 2), N)
    rm = np.hstack((0., -rm[::-1], rm))
    rm = np.unique(np.hstack((0, rm, coords[:, 0])))
    x = np.unique(np.hstack((xC - L/2, xC + 0.5 * (rm[:-1] + rm[1:]), xC + L/2)))
    y = np.unique(np.hstack((yC - L/2, yC + 0.5 * (rm[:-1] + rm[1:]), yC + L/2)))[::-1]
    
    layering = get_layering(Point(xC, yC), center=True)
    
    gr = Grid(x, y, layering['z'], axial=False)
    gr.layering = layering
    return gr

def grid_fr_polygons(pgons, L=15000., N=50):
    """Return grid from wells as specified in userinput.json.
    """
    coords = []
    for pgon in pgons:
        coords.append(pgon['geometry'].xy)
    coords = np.array(coords)       
    xC, yC = coords.mean(axis=0)
    
    r = np.logspace(np.log10(5), np.log10(L / 2), N)
    r = np.hstack((-r[::-1], r))
    x = np.unique(np.hstack((xC + r,       coords[:, 0])))
    y = np.unique(np.hstack((yC + r[::-1], coords[:, 1])))[::-1]
    
    layering = get_layering(Point(xC, yC), center=True)
    
    gr = Grid(x, y, layering['z'], axial=False)
    gr.layering = layering
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
        self.layering = get_layering(ctr, center=True)        
        self.gr = get_grid(ctr=ctr, wc=wc, w=w, N=N, z=self.layering['z'])
        self.set_screen(top=None, bot=None)
        self.set_IDOMAIN()
        self.set_model_params()        
        self.set_ih()
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
    
    def set_model_params(self, pnt):
        """Set model parameters (not the boundary conditions).
        
        These parameters should come from a database that yield them
        when given the coordinates of a point in Belgium in crs 31370
        
        """        
        gr = self.gr
        layering = self.gr.layering
        
        # --- Generate the model parameters (for the time being)
        self.kx = gr.const(layering['k'])
        self.ky = gr.const(layering['k'])
        self.kz = gr.const(layering['k33'])
        self.Ss = gr.const(layering['ss'])
        self.Ss[0] = layering['sy'][0]/ gr.dz[0]
        return None
    
    def set_IDOMAIN(self):
        self.IDOMAIN = self.gr.const(1, dtype=int)
        
    def set_ih(self):
        """Set initial head (all zero)"""
        self.HI = self.gr.const(0.)

    def set_wel(self, wells):
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
        
        fq = np.zeros(len(wells), dtype=dtypeQ)
        
        for i, well in enumerate(wells):
            xyz = np.array(well['geometry'].x, well['geometry'].y, gr.zm[0])
            idx = gr.Iglob_from_xyz(xyz)
            for flow_rate in well.flow_rates:
                t, Q = flow_rate
                self.WEL[t]
            fq['I'] = idx
            fq['q'] = 0.
        # hier moet nog de tijd overheen
            
            
            
            
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
            levels=np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
        levels.sort()
            
        
        if t is None:
            t = self.out['t'][-1]
        
        it = np.where(self.out['t'] <= t)[0][-1]
        
        # --- The extractions
        last_key = list(self.WEL.keys())[-1]
        Iwel = self.WEL[last_key]['I']
        Qwells = self.out['Q'][it-1].ravel()[Iwel].sum()
        #print(f"Qwells[t={self.out['t'][it]:.2f}] = {Qwells:8.2f} m3/d")

        if ax is None:
            title = f"Verlaging [m] at t={self.out['t'][it]:.0f} d after start extraction, Q={Qwells:.0f} m/d."
            ax = etc.newfig(title, 'x [m]', 'y [m]')
        
        # --- plot the surface water background (used in the model)
        self.clipped.plot(ax=ax, color='blue', linewidth=1)
        
        # --- contour the drawdown after the last time step
        phi = self.out['Phi'][it][0]
        gr = self.gr
        Cs = ax.contour(gr.xm, gr.ym, phi, colors='k', levels=levels)
        ax.clabel(Cs, inline=1, fontsize=10, fmt='%1.2f')
        
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'dd_example_1.png'))
        
        # --- water budget: all in the out dictionary
        print()
        print(f"Overall water budget during time step {it}, {self.out['t'][it-1]:.2f} <= t <= {self.out['t'][it]} d")
        print(f"Overall Q[{it}] ={self.out['Q'][it-1].sum():.2f} m3/d")        
        print()
        
        print(f"Individual budget items during time step {it}")
        
        # --- The extractions
        last_key = list(self.WEL.keys())[-1]
        Iwel = self.WEL[last_key]['I']
        Qwells = self.out['Q'][it-1].ravel()[Iwel].sum()
        print(f"Qwells[t={self.out['t'][it]:.2f}] = {Qwells:8.2f} m3/d")
        
        # --- Infiltration from surface water
        # self.out['Qriv']
        
        # --- Flow over the model boundary
        Qghb = self.out['Qghb'][it-1].sum()
        print(f"Qghb[  t={self.out['t'][it]:.2f}] = {Qghb:8.2f} m3/d")
        
        # --- Storage
        Qsto = self.out['Qs'][it-1].sum()
        print(f"Qsto[  t={self.out['t'][it]:.2f}] = {Qsto:8.2f} m3/d")
        
        # --- total water budget over time step
        Qtot = Qwells + Qghb + Qsto
        print()
        print(f"Overall water budet during time step {it}:")
        print(f"Qtot[{it}] = Qwwells[{it}] + Qghb[{it}] + Qsto[{it}]")
        print(f"{Qtot:8.2f} =    {Qwells:8.2f} + {Qghb:8.2f} + {Qsto:8.2f} m3/d")
        print()
        
        return None

# %%
ctr = (193919, 194774)

vtmdl = Vt_model(ctr, wc=5, w=15000, z=np.array([0, -30.]), N=100, Q=2400.)
vtmdl.run_mdl(t=np.logspace(-3, np.log10(180)))
vtmdl.evaluate(t=None, ax=None, levels=None)

plt.show()
print('Done!')
