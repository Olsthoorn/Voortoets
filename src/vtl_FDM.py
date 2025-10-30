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
import geopandas as gpd
import etc

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local / project
from src import vtl_surf_water as surf_wat
from fdm.src.fdm3t import fdm3t
from fdm.src.fdm3t import dtypeQ, dtypeH, dtypeGHB
import src.vtl_interventions as iv

# %%
class Vt_model():
    """Voortoets model, simple version, 1 layer FDM."""

    def __init__(self, case, wc=5, L=15000, t_end=365., tsmult=1.25):
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
        self.case = case
        
        # --- model grid from case['interventions']
        gr = iv.grid_from_interventions(case['interventions'], L=L,
                        tsmult=tsmult, show=False)
        self.gr = gr
        
        self.HI = gr.const(0.)
        self.idomain = gr.const(1, dtype=int)
        
        self.set_model_params()
        
        # ---- Boundary conditions from interventions
        interventions = case['interventions']
        if 'dewatering_polygon' in interventions:
            dw_pgons = interventions['dewatering_polygon']
            self.CHD = iv.chd_fr_dewatering_polygons(gr, dw_pgons)
            
        if 'hardening_polygon' in interventions:
            h_pgons = interventions['hardening_polygon']
            self.WEL = iv.wel_fr_hardening_polygons(gr, h_pgons)
            
        if 'extraction_general_point' in interventions:
            wells = interventions['extraction_general_point']
            self.WEL = iv.wel_fr_extraction_general_points(gr, wells)
            
        if 'extraction_irrigation_point' in interventions:
            wells = interventions['extraction_irrigation_point']
            self.WEL = iv.wel_fr_extraction_irrigation_points(gr, wells)
            
        if 'recharge_point' in interventions:
            rch_wells = interventions['recharge_point']
            self.WEL = iv.wel_fr_recharge_points(gr, rch_wells)
            
        if 'dewatering_line' in interventions:
            dw_lines = interventions['dewatering_line']
            self.CHD = iv.chd_fr_dwatering_lines(gr, dw_lines)
            
        # --- simulation time from interventions
        self.times = self.set_sim_times(tsmult=tsmult, t_end=t_end)
        
        # --- surface water from national Open Street Map
        self.GHB = self.set_surface_water_ghb(w=5, c=5)
        
        # --- convert time field to time_step
        self.convert_time_field_to_time_index()
        
        # --- put all model run parameters in a kwarg dict
        self.set_model_run_kwargs()
        
    
    def set_sim_times(self, tsmult=1.25, t_end=365.):
        """Return simulation time based on boundary times."""
        
        dt = np.ones(40) * 0.1
        for i in range(1, len(dt)):
            dt[i] = tsmult * dt[i-1]
        tr = np.hstack((0, np.cumsum(dt)))

        t_tuple = []
        if hasattr(self, 'WEL'):
            t_tuple.append(np.array(list(self.WEL.keys()), dtype=float))            
        if hasattr(self, 'CHD'):
            t_tuple.append(np.array(list(self.CHD.keys()), dtype=float))            
        if hasattr(self, 'GHB'):
            t_tuple.append(np.array(list(self.GHB.keys()), dtype=float))            
        t = np.unique(np.hstack(t_tuple))
        t = t[t <= t_end]
        
        times = [t[0]]
        for t1, t2 in zip(t[:-1], t[1:]):
            tt = t1 + tr
            times.append(tt[np.logical_and(tt >= t1, tt <= t2)])
        times = np.unique(np.hstack((*times, t_end)))
        self.times = times
        return times


    def convert_time_field_to_time_index(self):
        """Convert WEL, CHD and GHB to the form required by fdm3t"""
        
        times = self.times
        # --- fixed Q or wells
        if hasattr(self, 'WEL'):
            WEL = {}
            for t in self.WEL.keys():
                it = int(np.where(times == t)[0][0])
                WEL[it] = np.zeros(len(self.WEL[t]), dtype=dtypeQ)
                WEL[it]['I'] = self.WEL[t]['I']
                WEL[it]['q'] = self.WEL[t]['Q']
            self.WEL = WEL
                        
        # --- fixed heads
        if hasattr(self, 'CHD'):
            CHD = {}
            for t in self.CHD.keys():
                it = int(np.where(times == t)[0][0])
                CHD[it] = np.zeros(len(self.CHD[t]), dtype=dtypeH)
                CHD[it]['I'] = self.CHD[t]['I']
                CHD[it]['h'] = self.CHD[t]['h']                    
            self.CHD = CHD
            
        # --- general head boundaries
        if hasattr(self, 'GHB'):
            GHB = {}
            for t in self.GHB.keys():
                it = int(np.where(times == t)[0][0])
                GHB[it] = np.zeros(len(self.GHB[t]), dtype=dtypeGHB)
                GHB[it]['I'] = self.GHB[t]['I']
                GHB[it]['h'] = self.GHB[t]['h']
                GHB[it]['C'] = self.GHB[t]['C']                                  
            self.GHB = GHB
        
    def set_model_params(self):
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
        
    
    def set_model_run_kwargs(self):
        self.kwargs = {'gr': self.gr,                       
                  't': self.times,
                  'k': (self.kx, self.ky, self.kz),
                  'c': self.c if hasattr(self, 'c') else None,
                  'ss': self.Ss,
                  'fh':  self.CHD if hasattr(self, 'CHD') else None,
                  'ghb': self.GHB if hasattr(self, 'GHB') else None,
                  'fq' : self.WEL if hasattr(self, 'WEL') else None,
                  'hi' : self.HI,
                  'idomain': self.idomain,
        }
        

    def set_surface_water_ghb(self, w=5, c=5):
        """Set GHB to surface water from Open Street Map.
        
        Parameters
        ----------
        w: float
            with the all surface waters [m]
        c: float
            bottom resistance of all surface waters [d]
        """        
        gr = self.gr
        
        # --- get surface water length per model cell.
        self.clipped= surf_wat.clip_water_to_gr(gr)
        
        L = surf_wat.line_length_per_gr_cell(gr, self.clipped)['water_length_m']
                        
        ghb = np.zeros(gr.nx * gr.ny, dtype=dtypeGHB)
        ghb['I'] = gr.NOD[0].flatten()
        ghb['C'] = L * w / c
        ghb['h'] = 0.
        
        ghb = ghb[ghb['C'] > 0]
        t = 0.        
        return {t: ghb}


    def run_mdl(self):
        """Run the fdm3t code returning results in dict out and self.out."""      
        self.out = fdm3t(**self.kwargs)
        return self.out


    def evaluate(self, t=None, ax=None, levels=None):
        """Plot at t and return the model results for the user.
        
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
        if hasattr(self, 'WEL'):
            last_key = list(self.WEL.keys())[-1]
            Iwel = self.WEL[last_key]['I']
            Qwells = self.out['Q'][it-1].ravel()[Iwel].sum()
        #print(f"Qwells[t={self.out['t'][it]:.2f}] = {Qwells:8.2f} m3/d")

        if ax is None:
            title = f"{self.case['simulation_name']}: Verlaging [m] at t={self.out['t'][it]:.0f} d after start ingreep" #, Q={Qwells:.0f} m/d."
            ax = etc.newfig(title, 'x [m]', 'y [m]')
        
        # --- plot the surface water background (used in the model)
        self.clipped.plot(ax=ax, color='blue', linewidth=1)
        
        # --- contour the drawdown after the last time step
        phi = self.out['Phi'][it][0]
        gr = self.gr
        Cs = ax.contour(gr.xm, gr.ym, phi, colors='k', levels=levels)
        ax.clabel(Cs, inline=1, fontsize=10, fmt='%1.2f')
        
        # --- plot the intervention contours and locations
        interventions = self.case['interventions']
        for k in interventions:
            for geob in interventions[k]:
                gpd.GeoSeries([geob['geometry']]).plot(
                    ax=ax, edgecolor='r', facecolor='none', linewidth=1)
        
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'dd_example_1.png'))
        
        # --- water budget: all in the out dictionary
        print()
        print(f"Overall water budget during time step {it}, {self.out['t'][it-1]:.2f} <= t <= {self.out['t'][it]} d")
        print(f"Overall Q[{it}] ={self.out['Q'][it-1].sum():.2f} m3/d")        
        print()
        
        print(f"Individual budget items during time step {it}")
        
        # --- The extractions
        if hasattr(self, 'WEL'):
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

def zoom(zoom_factor):
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx, dy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

    ax.set_xlim(xc - dx / zoom_factor, xc + dx / zoom_factor)
    ax.set_ylim(yc - dy / zoom_factor, yc + dy / zoom_factor)
    return(ax)
    
# %%
project_folder = os.path.join(os.getcwd(), 'data/6194_GWS_testen/')
if not os.path.isdir(project_folder):
    project_folder = os.path.join(os.getcwd(), '../data/6194_GWS_testen/')
    assert os.path.isdir(project_folder), f"No folder <{project_folder}>"


cases = iv.cases_fr_json(project_folder)

case = cases[4]
case = cases[0]

vtmdl = Vt_model(case, wc=5, L=15000, t_end=365., tsmult=1.25)

vtmdl.run_mdl()
vtmdl.evaluate(t=183, ax=None, levels=None)
zoom(1.)

plt.show()
print('Done!')
