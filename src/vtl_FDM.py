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

from scipy.interpolate import RegularGridInterpolator

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
def zoom(zoom_factor):
    """Zoom in on curent axes by zoom_factor."""
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx, dy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

    ax.set_xlim(xc - dx / zoom_factor, xc + dx / zoom_factor)
    ax.set_ylim(yc - dy / zoom_factor, yc + dy / zoom_factor)
    return(ax)


def bilinear_interpolate_stack(gr, h, xp, yp):
    """
    Bilinear interpolation for a stack of 2D arrays h(t, y, x)
    at a single point (xp, yp).
    """    
    # find indices of cell lower-left corner
    ix = np.searchsorted(gr.xm, xp) - 1
    iy = np.searchsorted(gr.ym, yp) - 1
    
    # clip to valid range
    ix = np.clip(ix, 0, gr.nx - 2)
    iy = np.clip(iy, 0, gr.ny - 2)
    
    # fractional position inside the cell
    x1, x2 = gr.xm[ix], gr.xm[ix+1]
    y1, y2 = gr.ym[iy], gr.ym[iy+1]
    tx = (xp - x1) / (x2 - x1)
    ty = (yp - y1) / (y2 - y1)
    
    # corner values (broadcasts over time dimension)
    f11 = h[:, iy, ix]
    f21 = h[:, iy, ix+1]
    f12 = h[:, iy+1, ix]
    f22 = h[:, iy+1, ix+1]
    
    # bilinear interpolation
    return ((1-tx) * (1-ty) * f11 +
                tx * (1-ty) * f21 +
                (1-tx) * ty * f12 +
                    tx * ty * f22)


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
            self.CHD = iv.chd_fr_dewatering_polygons(gr, dw_pgons, t_end=t_end)
            
        # --- The items yielding self.WEL should be mutually exclusive
        # TODO enforce this
        if 'hardening_polygon' in interventions:
            # Interpreted as extracting a percentage of the net recharge.
            h_pgons = interventions['hardening_polygon']
            self.WEL = iv.wel_fr_hardening_polygons(
                gr, h_pgons, recharge=0.001, t_end=t_end)
            
        if 'extraction_general_point' in interventions:
            wells = interventions['extraction_general_point']
            self.WEL = iv.wel_fr_extraction_general_points(gr, wells, t_end=t_end)
            
        if 'extraction_irrigation_point' in interventions:
            wells = interventions['extraction_irrigation_point']
            self.WEL = iv.wel_fr_extraction_irrigation_points(gr, wells, t_end=t_end)
            
        if 'recharge_point' in interventions:
            rch_wells = interventions['recharge_point']
            self.WEL = iv.wel_fr_recharge_points(gr, rch_wells, t_end=t_end)
            
        if 'dewatering_line' in interventions:
            dw_lines = interventions['dewatering_line']
            self.CHD = iv.chd_fr_dwatering_lines(gr, dw_lines, t_end=t_end)
            
        # --- simulation time from interventions
        self.times = self.set_sim_times(tsmult=tsmult, t_end=t_end)
        
        # --- surface water from national Open Street Map
        self.GHB = self.set_surface_water_ghb(w=5, c=5)
        # TODO Temporarily        
        self.GHB = {0.: np.zeros(1, dtype=dtypeGHB)}
        
        
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
        t = np.unique(np.hstack((0., *t_tuple, t_end)))
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
        images_dir = os.path.join(os.getcwd(), 'images/vtl_fdm')
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(os.getcwd(), '../images/vtl_fdm')
        
        if levels is None:
            levels=np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
            levels = np.unique(np.hstack((-levels, levels)))
        
        case_name = self.case['simulation_name']
        inv_list = list(self.case['interventions'].keys())
        invs = f" interventions: [{', '.join(inv_list)}]"
    
        if t is None:
            t = self.out['t'][-1]
        
        it = np.where(self.out['t'] <= t)[0][-1]
        
        if ax is None:
            title = (f"{case_name} " + invs + ', ' +
                     f"DDN [m] at t={self.out['t'][it]:.0f}" + "\n"
                     f"Qfq = {self.out['Qfq'][it].sum():.3g}, " +
                     f"Qfh = {self.out['Qfh'][it].sum():.3g}, " +
                     f"Qghb= {self.out['Qghb'][it].sum():.3g}, " +
                     f"Qs= {self.out['Qs'][it].sum():.3g} m3/d")
            ax = etc.newfig(title, "xB [m]", "yB [m]", figsize=(10, 10))
        
        # --- plot the surface water background (used in the model)
        self.clipped.plot(ax=ax, color='blue', linewidth=1)
        
        ax.set_xlim(self.gr.x.mean() - 1500. , self.gr.x.mean() + 1500.)
        ax.set_ylim(self.gr.y.mean() - 1500. , self.gr.y.mean() + 1500.)
        
        # --- contour the drawdown after the last time step
        phi = self.out['Phi'][it][0]
        gr = self.gr
        Cs = ax.contour(gr.xm, gr.ym, phi, colors='k', levels=levels)
        ax.clabel(Cs, inline=True, levels=Cs.levels, fontsize=10, fmt='%.2f')
        
        # --- plot the intervention contours and locations
        interventions = self.case['interventions']
        for k in interventions:
            for geob in interventions[k]:
                gpd.GeoSeries([geob['geometry']]).plot(
                    ax=ax, edgecolor='r', facecolor='none', linewidth=1)
        
        # ax.figure.savefig(os.path.join(images_dir, f"case_name_t{t:.0f}d.png"))
        
        # --- zoom in and save again
        # zoom(10)
        
        ax.figure.savefig(os.path.join(images_dir, f"case_name_t{t:.0f}d_Z10.png"))
                
        # --- water budget: all in the out dictionary        
        o = self.out
        txt = []
        txt.append("Model water budget\n------------------")
        txt.append(f"{'it':>3}{'t1':>7}{'t2':>7}" +
              f"{'Qfh':>8} {'Qfq':>8} {'Qs':>8} {'Qghb':>8} {'Q':>8}")
        for it in range(1, len(o['t']) - 1):
            t1, t2 = o['t'][it - 1], o['t'][it]
            txt.append(f"{it:3} {t1:6.1f} {t2:6.1f} " +
                  f"{o['Qfh'][it].sum():8.0f} " +
                  f"{o['Qfq'][it].sum():8.0f} " +
                  f"{o['Qs'][ it].sum():8.0f} " +
                  f"{o['Qghb'][ it].sum():8.0f} " +
                  f"{o['Q'][  it].sum():8.0f}"
                  )
        fname = os.path.join(images_dir, f"{case_name}_Qt.txt")
        with open(fname, 'w') as f:
            f.write('\n'.join(txt) + '\n')
        
        for s in txt:
            print(s)
        print()
        
        # --- Plot the water budget components
        title = f"case {case_name}: Budget components Qfh, Qfq, Qghb, Qs [all m3/d]"
        ax = etc.newfig(title, 't [d]', 'Q [m3/d]', figsize=(10, 6))
        ax.plot(o['t'][1:],
                o['Qfh'].sum(axis=-1).sum(axis=-1).sum(axis=1),
                label='Qfh')
        ax.plot(o['t'][1:],
                o['Qfq'].sum(axis=-1).sum(axis=-1).sum(axis=1),
                label='Qfq')
        ax.plot(o['t'][1:],
                o['Qghb'].sum(axis=-1).sum(axis=-1).sum(axis=1),
                label='Qghb')
        ax.plot(o['t'][1:],
                o['Qs'].sum(axis=-1).sum(axis=-1).sum(axis=1),
                label='Qs')
        
        ax.legend(loc='best')
        
        ax.figure.savefig(os.path.join(images_dir, f"{case_name}_Qt.png"))
        
        # --- Plot h(t) at the reception point
        name = self.case['receptors']['receptor_point'][0]['name']
        recp = self.case['receptors']['receptor_point'][0]['geometry']
        
        title = f"Case {case_name}: Verandering grwst in reception points"
        ax = etc.newfig(title, 't [d]', 'Verandering grwst [m]', figsize=(10, 6))

        for ilay in range(gr.nlay):
            phi_lay = o['Phi'][:, ilay, : ,:]
            h_recp = bilinear_interpolate_stack(gr, phi_lay, recp.x, recp.y)
            ax.plot(o['t'], h_recp, 
                    label=f"{name} laag {ilay} z=[{gr.z[ilay]:.1f} - {gr.z[ilay + 1]:.1f}] at xy = ({recp.x:.0f}, {recp.y:.0f})")
        ax.legend(loc='best')
        
        ax.figure.savefig(os.path.join(images_dir, f"{case_name}_ht.png"))
        
        return None
    
# %%
project_folder = os.path.join(os.getcwd(), 'data/6194_GWS_testen/')
if not os.path.isdir(project_folder):
    project_folder = os.path.join(os.getcwd(), '../data/6194_GWS_testen/')
    assert os.path.isdir(project_folder), f"No folder <{project_folder}>"


cases = iv.cases_fr_json(project_folder)

case = cases[0]
case = cases[1]
# case = cases[2]
# case = cases[3]
# case = cases[4]
# case = cases[5]
# case = cases[6] # retourputten
# case = cases[7] # retourputten
# case = cases[8]
# case = cases[9]
# case = cases[10]


vtmdl = Vt_model(case, wc=5, L=15000, t_end=365., tsmult=1.25)

vtmdl.run_mdl()
vtmdl.evaluate(t=183, ax=None, levels=None)

plt.show()
print('Done!')

# %%
