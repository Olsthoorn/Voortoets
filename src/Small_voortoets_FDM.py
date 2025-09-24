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
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import etc                    # My module under tools/etc
from fdm import Grid, fdm3t   # My fdm module under tools/fdm

dirs = etc.Dirs(os.getcwd())  # My module dirs sets project directories

if "" not in sys.path:
    sys.path.insert(0, "")

# %%

class vt_model():
    """Voortoets model, simple version, 1 layer FDM."""

    def __init__(self, xy_ctr, R=7500, dx=5, nmax=200):
        """
        Parameters
        ----------
        xy_ctr: (float, float)
            coordinates of model center (center of center cell)
        R: float
            model radius (i.e. half width and half height)
        dx: float        
            minimum values of dx and of dy at self.xy_ctr.
        a: float   1 < a < 2)
            desired multiplier for dx and dy
        nmax: int
            maximum number of cells in either x and y directions
        """
        
        if not iter(xy_ctr) or not len(xy_ctr) == 2:
            return TypeError("xy_ctr must be an interable of length 2")                             
        
        self.xy_ctr = xy_ctr
        self.xGr, self.yGr = self.get_grid_coordinates(R, dx, nmax)
        
        
    def get_grid_coordinates(self, R, dx=5, a=1.1, nmax=200):
        """Return generated grid line coordinates around self.xy_ctr.
        """
        for n in range(nmax // 2):
            _x = np.logspace(np.log10(dx/2), np.log10(R), n)
            a = _x[1] / _x[0]
            if a <= 1.1:
                break
        _x = np.hstack((-_x[::-1], _x))
        
        xc, yc = self.xy_ctr
        xGr = xc + _x
        yGr = yc + _x[::-1]

        return xGr, yGr
    
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
        return surfwater(self.xy_ctr, self.R)
    
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
        gr = Grid(self.xGr, self.yGr, self.zGr, min_dz= 5, axial=False)
        self.k (gr.const(self.kh), gr.const(self.kh), gr.const(self.kv))
        S  = gr.const(Ss=self.Ss)
        S[0] = self.Sy / gr.DZ[0]
        self.S = S
        self.gr = gr
        return None
    
    def boundary_conditions(self):
        self.IDOMAIN = self.gr.const(1, dtype=int)
        self.IH = self.gr.const(0.)

    def wells(self, xyQ):
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
        self.WEL = self.xyQ2wel(xyQ)
        
    def make_riv(self, xyChb):
        """Return RIV.
        
        The surface water data are used so generate
        (iz, iy, ix, C, h, hriv)  values per stress period.
        In this model, the h-values are all zoer
        and remain constant throughout the model run.
        
        The length of the surface water segments per cell
        crossed are obtained by intersecting the surface
        water line segments with the grid.        
        """
        # --- Convert surface water to model input.        
        self.RIV = self.convert_surface_water_to_RIV()
       
    def make_ghb(self, xyCh):
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
        self.GHB = self.convert_surface_water_to_GHB()

       
    def make_chd(self, xyH):
        """Return CHD.
        
        Return a CHD object with fields
        iz, iy, ix, h
        telling which nodes have constant head.
        This module can be used to set afeal
        drawdown or surface water changes.
        self.chd = set_chd()

        """ 
        self.chd = self.convert_chd_area_to_CHD()
        
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
