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
from fdm.src.fdm3t import Fdm3t, fdm3t  # My fdm module under tools/fdm

# import matplotlib.pyplot as plt
# --- Local / project
from fdm.src.mfgrid import Grid
from fdm.src.fdm3t import Fdm3t, fdm3t
from fdm.src.fdm3t import dtypeQ, dtypeH, dtypeGHB

dtypeQ   = np.dtype([('I', np.int32), ('q', float)])
dtypeH   = np.dtype([('I', np.int32), ('h', float)])
dtypeGHB = np.dtype([('I', np.int32), ('h', float), ('C', float)])

from src.select_surf_wat_15x15km import get_water_length_per_cell


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
    xi = np.logspace(np.log10(wc / 2), np.log10(w / 2), int(N % 2))
    
    # --- real world coordinates of vertical grid lines
    x = x0 + np.hstack((-xi[::-1], xi))
    
    # --- real world coordinaters of horizontal grid lines
    y = y0 + np.hstack((xi[::-1], -xi))
    
    # --- create grid object
    gr = Grid(x, y, z, axial=False)
    return gr


class vt_model():
    """Voortoets model, simple version, 1 layer FDM."""

    def __init__(self, ctr, wc=5, w=15000, z=np.array([0, -30.]), N=100):
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
    
    def get_layer_buildup(self, database):
        """Return layer definition at xy_ctr obtained from database.
        
        Parameters
        ----------
        databse: object
            An object representing the layers from which
            the layers at some x, y coordinate can be extracted.
            The extraction likely also has the hydraulic layer properties,
            which will be used in the model.
        """
        # --- it's not known yet how to do this (must wait for example database)
        
        # --- Bottom elevation of phreatic aquifer relative to ground surface
        self.D = database.z0(self.xy_ctr) - database.D(self.xy_ctr)
        
        # --- Layer hydraulic parameters
        self.kh = database.kh(self.xy_ctr)
        self.kv = database.kv(self.xy_ctr)
        self.Ss = database.Ss(self.xy_ctr)
        self.Sy = database.Sy(self.xy_ctr)
        
        # --- may return None instead
        return database(self.xy_ctr)
    
    def get_surf_water(self, surfwater):
        """Return the surface water around xy_ctr within radius R.
        
        It's assumed that the surfwater database is like a shapefile
        representing line segments with fields like x1, y1, x2 y2, width, resistance.
        
        The size of the area to sample depends on the closest segment coordinate
        lying within radius R
        """
        # --- Must await an example database before this can be implemented
        # return surfwater(self.xy_ctr, self.R)
    
    def get_filter_elevation(self, screen_top=None, screen_bot=None):
        """Return elevation of top and bottom of well screen.
        
        Parameters
        ----------
        screen_top: float | None
            top of well screen [m relative to groound sruface i.e. [<=0]
            Default = None --> ground surface.
        screen_bot: float | None 
            bot of well screen [m relative to ground surface [<0]]
            Default = None -->  base of phreatic aquifer.
        """
        if screen_top > 0 or screen_bot > 0:
            raise ValueError("screen_top and screen_top must both be < zero")
        if screen_bot > screen_top:
            screen_top, screen_bot = screen_bot, screen_top
        
        zGr = [0]    
        if screen_top < -5:
            zGr.append(screen_top)
        if screen_bot > self.D + 5:
            zGr.append(screen_bot)
        zGr.append(self.D)
        self.zGr = zGr
        return None
    
    def make_model(self):
        """Return the model without its boundary conditions."""
        # --- Generate the model grid
        self.k (self.gr.const(self.kh),
                self.gr.const(self.kh),
                self.gr.const(self.kv))
        S  = self.gr.const(Ss=self.Ss)
        S[0] = self.Sy / gr.DZ[0]
        self.S = S        
        return None
    
    def boundary_conditions(self):
        self.IDOMAIN = self.gr.const(1, dtype=int)
        self.IH = self.gr.const(0.)

    def get_fq(self, xyQ):
        """Return well boundary conditions.
        
        Parameters
        ----------
        wells: pd.DataFrame
            fields "x, y, Q, Q, Q, Q, Q, Q"
            x, y are well coordinates, Q is month value, maximum 6 values.
           
            If only one Q is given, this Q if for the entire period of 180 days.
            Else values pertain to successive months. The extraction then stops
            after the last Q was processed.
            
            The index of the DataFrame would be the well identifyer.
            Other layouts are possible, but this one is really simple.
        """  
        # --- Convert to model input
        x, y, Q = xyQ   
        I = gr.Iglob_from_xyz(x, y, gr.zm[0])
        fq = np.zeros(len(I), dtype=dtypeQ)
        fq['I'] = I
        fq['q'] = Q
        self.fq = {0: fq}
        
        
    def get_ghb(self, w=5, c=5):
        """Return RIV.
        
        GHB are best used here to get the flow across the
        model boundary. Although this can also be done using
        CHD which then can be required to meet a criterion
        telling that the flow crossing the boundary is sufficiently
        small to ignore it.
        
        An advantage of using GHB for the boundary is that it can emulate
        the flow across radius R in the Hantush well function.       
        """
        # --- Convert surface water to model input.
        grid = get_water_length_per_cell(gr, show=False)
        GHB = np.zeros(gr.nod, dtype=self.dtypeGHB)
        GHB['I'] = gr.NOD.flatten()
        GHB['C'] = grid['water_length_m'] * w / c
        GHB['h'] = 0.
        GHB = GHB[GHB['C'] > 0]
        self.GHB = {0: GHB}

       
    def get_chd(self, xyH):
        """Return CHD.
        
        Return a CHD object with fields
        iz, iy, ix, h
        telling which nodes have constant head.
        This module can be used to set afeal
        drawdown or surface water changes.
        self.chd = set_chd()

        """
        x, y, h = xyH
        I = gr.Iglob_from_xyz(x, y, gr.zm[0])
        fh = np.zeros(len(I), dtype=dtypeH)
        fh['I'] = I
        fh['h'] = h
        self.fq = {0: fh}
        
        
    def run_mdl(self, t=None):        
        out = fdm3t(gr=self.gr, k=(self.kx, self.ky, self.kz), t=t,
                    S=self.S, IH=self.IH, idomain=self.IDOMAIN,
                    chd=self.CHD, wel=self.WEL)
        return out
        
    def evaluate(self, t=None, ax=None,
            levels=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]):
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
        it = np.where(self.out.t <= t)[0][-1]
        if ax is None:
            title = f"Verlaging [m] at t={self.out.t[it]} d after start extraction."
            ax = etc.newfig(title, 'x [m]', 'y [m]')
        
        phi = self.out.Phi[it]
        Cs = ax.contour(self.gr.X, self.gr.Y, phi, levels=self.levels)
        ax.label(Cs, ...)
        
        # --- water budget: all in the out dictionary
        self.out.Q
        
        # --- The extractions
        self.out.Qwel
        
        # --- Infiltration from surface water
        self.out.Qiv
        
        # --- Flow over the model boundary
        self.out.Qghb
        
        # --- Storage
        self.out.Qsto
        
        return None

# %%
