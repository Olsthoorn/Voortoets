# Hantush als compromis voor de berekeing van de verlaging van freatische grondwater

# %%
import os, sys
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import k0 as K0
from tools.fdm.src.fdm3 import fdm3
from tools.fdm.src.mfgrid import Grid
from tools.etc.etc import logo

import vtl_surf_water as sw

cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT-Voortoets' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT-Voortoets") + 1], "Coding", "images")

print(sys.executable)

# %%
def sinspace(x1, x2, th1=0, th2=np.pi, N=100):
    """Return array of points between x1 and x2 distributed like sine function.
    
    x1, x2: float
        Range limits
    th1, th2: floats
        Starting and finish angle determining first step size.
    N: int
        Number of points to be generated
    """
    ds = np.sin(np.linspace(th1, th2, N))
    x = x1 + (x2 - x1) * np.hstack((0, np.cumsum(ds) / ds.sum()))
    return x

def concat_space(x, N):
    """Return new array that repeats array x N time."""
    x_ = x
    for i in range(N-1):
        x = np.hstack((x[:-1], x[-1] + x_))
    return x

def two_simultaneous_xsections(L=500, D=25, k=1, ws=[0.25, 0.5, 1, 2], xw=2500, Qw=None, rDitch=1):
    """Plot results of two parallel cross sections.
    
    Simulate two 1D cross section models in a single run, ones single 2-row model,
    each model in one row.
    
    One model has individual ditches.
    The othe rmodel has an equivalent semi-permeable top layer.
    
    Parameters
    ----------
    L: float
        Distance between the ditches.
    D: float
        Aquifer thickness.
    k: float
        Horizontal hydraulic conductivity.
    ws: Iterable
        Ditch resistances [d/m]
    xw: float
        Location of well
    Qw: float
        Well flow m2/d
    rDitch: float
        Ditch radius (not relevant in 1D models)
    """
    # --- The grid: Two layers, two rows
    a =0.01 # --- small starting angle used in sinspace
    x = sinspace(x1=0, x2=500, th1=np.pi *a, th2=np.pi * (1-a), N=25)
    x = concat_space(x, 20)
    z = np.array([0, -2, -2-D])

    # --- Two rows, second ditch_mdl, ditch_mdl semi-confined
    gr = Grid(x - x.mean(), y=np.array([-1, 0, 1]), z=z)

    # --- Get the locations of the ditches in the the ditch_mdl?
    id = [int(p) for p in np.where(np.diff(x) < np.diff(x).min()+0.1)[0]]
    od =[id[0]]
    for p in id[1:]:
        if p != od[-1] + 1:
            od.append(p)

    # --- The well
    iw = gr.ix(xw)['idx'][0]
    
    # --- Model arrays
    kx, ky, kz = gr.const(k), gr.const(1e-20), gr.const(k)

    fig, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(10, 11))
    
    fig.suptitle(
        f"{2*L} m wide 1D-X-section with a line well\n"
        "Blue: recharge via ditch_mdl. Red: recharge via resistance top layer.\n"
        fr"Extraction $Q_w$={Qw} m2/d at $x$={gr.xm[iw]:.0f}" + f", $kD$={k * D} m2/d",
        fontsize=14)

    # --- Compute and plot model for each ditch resistance w
    for w, ax in zip(ws, axs.ravel()):
        clrs = cycle('brgkmc')
        # --- Radiale weerstand [d/m]
        if w is None:
            w = 1 / (np.pi * k) * np.log(D / (np.pi * rDitch))
        
        # --- Weerstand tweede doorsnede compatibel met die van de sloten [d]
        c = w * L

        kx[0] = 1e-20        
        kz[0, 0, :] = 1e-20 # --- Start making kz 0
        
        # --- In Xsec with ditch_mdl, computed adapt kz in both layers to match the given resistance
        Dx = gr.dx[od]
        w2 = gr.DZ[1, 0, od] / (2*Dx * kz[1, 0, od])  # w of ditch_mdl layer
        w1 = w - w2                                   # w of top layer
        w1[np.isclose(w1,0)] = 1e-6                   # prevend division by zero
        kz[0, 0, od] = gr.DZ[0, 0, od] / (2*Dx * w1)  # matching kz in top layer
        
        # --- Alternatively use GHB
        mask = gr.const(0, dtype=bool)
        mask[0, 0] = True
        GHB = gr.GHB(mask, 0., gr.DY[mask] / w)
                
        # --- Set kz in Xsec with resistance layer
        kz[0, 1, :] = gr.DZ[0, 1, :] / (2 * c)

        HI = gr.const(0.)
        FQ = gr.const(0.)

        FQ[1, :, iw] = Qw # --- Both rows (models) get the same well

        # --- Boundary array
        IBOUND = gr.const(1, dtype=int)        
        IBOUND[0, 0, od] = -1 # Dsn 1: Cellen in sloten fixed head.
        IBOUND[0, 1,  :] = -1 # Dsn 2: Cellen in de toplaag fixed head

        # ---Simulate
        out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB)

        # --- Analytic simulation
        # --- Xsec 1, with ditch_mdl
        kD = k*D
        phi = out['Phi'][1, 0]
        Qx  = out['Qx'][1, 0]
        phiX1 = np.zeros_like(phi); phiX1[0] = phi[0]
        phiX1[od[1:]] = phi[od[:-1]] - Qx[np.array(od[:-1]) + 10] / kD * (gr.xm[od[1:]] - gr.xm[od[:-1]])        
        
        # --- Xsec2, resistance layer on top
        lam = np.sqrt(kD*c)
        s2 = lam/kD * Qw/2 * np.exp(-np.abs(gr.xm-xw)/lam)

        # --- Show        
        title=(fr"$w$={w:.3g} d/m, $c$={c:.3g} d, $\lambda$={lam:.3g} m")        
        ax.set(title=title, xlabel=r"$x$[m]", ylabel=r"head $\phi$ [m]")

        for iL in range(1, gr.nz):
            clr = next(clrs)
            ax.plot(gr.xm, out['Phi'][iL, 0, :], color=clr, label=f'layer[{iL}], with ditch_mdl')
            clr = next(clrs)
            ax.plot(gr.xm, out['Phi'][iL, 1, :], color=clr, label=f'layer[{iL}], semi-confined')
            
        clr = next(clrs)
        ax.plot(gr.xm[od], phiX1[od], 's', ms=4, mfc='none', mec=clr, label='With ditch_mdl, analytic')  
        ax.plot(gr.xm[:], s2[:],  '.', mfc='none', mec=clr, label='With separating layer anlaytic')
            
        ax.plot(gr.xm[od],gr.xm[od] * 0, 'v', ms=6, mec='b', mfc='none', label='ditch locations')
        xl = np.ceil((3 * lam) / 100) * 100
        ax.set_xlim(-xl, xl)

        ax.grid()
        ax.legend(loc='lower left')
    
    logo(fig, os.path.basename(__file__))
    fig.savefig(os.path.join(images, "drawdown_xsec_4xw.png"))
    
    print(fr"lam/L = {np.sqrt(kD * np.array(ws) / L)}")
    return out
        
def two_simultaneous_3Dmodels(L=500, dxy=25, k=10, D=20, ws=[0.25, 0.5, 1, 2], xw=2500, Qw=-2400, rDitch=1):
    """Generate and run two jumeleaux (back-to-back) 3D (two-layer) FDM models.

    The models are separated from each other by a grid row
    between y=-0.001 and y=+0.001 that is given zero conductivity.    
    
    There is one well, allowing to simulate only one halfspace in each model.
    
    The second model,  called ditch_mdl, has parallel ditches that are L apart.
    The ditch_mdl model has an equivalent resistance layer on top.
    
    Parameters
    ----------
    L: float
        Distance between the ditches.
    dxy: float
        Grid step size [m].
    k: float
        Hydraulic conductivity of the aquifer.
    D: float
        Thickness of the aquifer.
    ws: Iterable
        tuple of ditich resitances (radial resistances) [d/m]
    xw: float
        Location of the well along the x-axis.
    Qw: float
        Well flow (extraction negative) [m3/d]
    rDitch: float
        Ditch radius    
    """
    # --- The grid
    a = 0.01 # --- small value used in sinspace determining first step size.
    x = sinspace(x1=0, x2=500, th1=np.pi *a, th2=np.pi * (1-a), N=25)
    x = concat_space(x, 20) # --- Repeat array x N times into a new array
    z = np.array([0, -2, -2-D])
    y= np.logspace(0, np.log10(5000), 51)
    y = np.hstack((-y[::-1], -0.001, 0.001, y))
    
    # --- Two rows, second ditch_mdl, ditch_mdl semi-confined
    gr = Grid(x - x.mean(), y=y, z=z)
    
    # --- Masks to select parts of the model
    ditch_mdl_rows  = gr.ym > 0   # --- The model with the ditches at the top
    clayer_mdl_rows = gr.ym < 0 # --- The Model with the resitance top layer
    
    wall = np.logical_and(np.logical_not(clayer_mdl_rows), np.logical_not(ditch_mdl_rows)) # The row in between the two models

    # --- At which ix sit the ditches in the ditch model?
    id = [int(p) for p in np.where(np.diff(x) < np.diff(x).min()+0.1)[0]]
    od =[id[0]]
    for p in id[1:]:
        if p != od[-1] + 1:
            od.append(p)

    # --- The well
    iw = gr.ix(xw)['idx'][0]    
    iwall = np.argmin(np.abs(gr.ym))
    iyw = [iwall-1, iwall+1]
    
    # --- Model arrays
    kx, ky, kz = gr.const(k), gr.const(k), gr.const(k)
    
    # --- Impervious wall between the two models
    # ky[:, wall, :] = 1e-20 # We set IBOUND[:, wall, :] to inactive below

    suptitle = (
        f"{gr.dx.sum():.0f} m wide and long 3D twin models each with a single well\n"
        "Blue: recharge via ditches. Red: recharge via resistance top layer.\n"
        fr"$Q_w$={Qw} m3/d at $(x,y)$=({gr.xm[iw]:.0f},0)" + f", $kD$={k * D} m2/d"
    )
    
    
    
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(10, 11))
    fig.suptitle(suptitle, fontsize=14)

    for w, ax in zip(ws, axs.ravel()): # w is ditch resistance [d/m]        
        # --- Radial resistance [d/m]
        if w is None:
            w = 1 / (np.pi * k) * np.log(D / (np.pi * rDitch))
        
        # --- Equivalent resistance layer compatible with the ditch_mdl [d]
        c = w * L

        kx[0] = 1e-20 # --- Start makking kx[0] zero
        kz[0] = 1e-20 # --- Start making kz[0] zero
        
        # --- In model with ditch_mdl, adapt kz to match the given ditch resistance
        # ditch_od = np.ix_(ditch_mdl, od)        
        # w2 = gr.DZ[1][ditch_od] / (2* gr.DX[1][ditch_od] * kz[1][ditch_od])  # w of ditch_mdl layer [d/m2]
        # w1 = w - w2                                   # w of top layer
        # w1[np.isclose(w1,0)] = 1e-6                   # prevent division by zero
        # kz[0][ditch_od] = gr.DZ[0][ditch_od] / (2*gr.DX[0][ditch_od] * w1)  # matching kz in top layer
        
        # --- Alternatively use DRN boundary condition
        mask = gr.const(0, dtype=bool)
        mask[1, :, od] = True
        mask[:, clayer_mdl_rows, :] = False
        GHB = gr.GHB(mask, 0., gr.DY[mask] / w)
        
        # --- Set kz in Xsec with resistance layer
        kz[0, clayer_mdl_rows, :] = gr.DZ[0, clayer_mdl_rows, :] / (2 * c)

        HI = gr.const(0.)
        FQ = gr.const(0.)

        FQ[1, iyw, iw] = Qw / 2 # --- Both rows (models) get the same well

        # --- Boundary array
        IBOUND = gr.const(1, dtype=int)        
        IBOUND[0, ditch_mdl_rows,  :] =  0 # Model 1: with cells not used because GHB are used
        IBOUND[0, clayer_mdl_rows, :] = -1 # Model 2: with equivalent resistance layer.
        IBOUND[:, wall, :] = 0
        
        # ---Simulate
        out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB)

        # --- Analytic simulation
        kD = k*D
        phi = out['Phi'][1, iyw, :]
        lam = np.sqrt(kD*c)
        
        s2 = Qw/(2 * np.pi* kD) * K0(np.abs(gr.xm - xw)/lam)

        # --- Show        
        title=(fr"$w$={w:.3g} d/m, $c$={c:.3g} d, $\lambda$={lam:.3g} m")        
        ax.set(title=title, xlabel=r"$x$[m]", ylabel=r"head $\phi$ [m]")

        for iL in range(1, gr.nz):            
            ax.plot(gr.xm, out['Phi'][iL, iyw[0], :], color='b', label=f'layer[{iL}], with ditch_mdl')            
            ax.plot(gr.xm, out['Phi'][iL, iyw[1], :], color='r', lw=0.5, label=f'layer[{iL}], semi-confined')
                    
        ax.plot(gr.xm[:], s2[:],  '.', mfc='none', mec='r', ms=8, label='Semi confined anlytic')
            
        ax.plot(gr.xm[od],gr.xm[od] * 0, 'v', ms=6, mec='b', mfc='none', label='ditch locations')
        xl = np.ceil((3 * lam) / 100) * 100
        ax.set_xlim(-xl, xl)        

        ax.grid()
        ax.legend(loc='lower left')
    logo(fig, os.path.basename(__file__))
    fig.savefig(os.path.join(images, "drawdown_3D_4xw.png"))
    
    # --- Contour the heads    
    fig, ax = plt.subplots(figsize=(10,9))
    fig.suptitle(suptitle, fontsize=14)    
    ax.set(title="Contours", xlabel="x", ylabel="y", xlim=(-2000, 2000), ylim=(-2000,2000))
    ax.grid()
    levels = np.array([-0.05, -0.1, -0.2, -0.5, -1., -2, -5.])[::-1]
    Cs = ax.contour(gr.xm, gr.ym, out['Phi'][-1], linewidths=0.5, levels=levels, colors='blue')
    ax.clabel(Cs, levels=Cs.levels, fmt='%.2f')
    for id in od:
        ax.vlines(gr.xm[od], ymin=0, ymax=2000, color='r', lw=2)        
    ax.plot(0, 0, 'ko')
    logo(fig, os.path.basename(__file__))
    fig.savefig(os.path.join(images, "contours-DeGlee-vs-Model.pdf"))
    
    print(fr"lam/L = {np.sqrt(kD * np.array(ws) / L)}")


def fdm_real_surfwater(xy=None, Q=-3000, b=500, Lmdl=15000, D=20, k=10, dxy=50, z=[0, -2, -22], w=2):
    r"""Generate and run a $L \times L$ model with extraction Q, in location xy. 
    
    The model has two layers one to host the wells and and for the regional aquifer.
    The radial surface water resistance is w [d/m]
    The surface water is obtained from the OSM of Belgium.
    
    Parameters
    ----------
    xy: tuple of floats
        Real-world coordinates of the well
    Q: float
        Well flow.
    b: float
        Distance between the ditches.
    Lmdl: float
        Width and height of the map of the model grid.
    D: float
        Thickness of the aquifer.
    k: float
        Hydraulic condutivity of the aquifer.
    dxy: float
        Grid step size.
    z: sequence of float
        Elevation of ground surface followed from layer bottom elevations.
    w: float
        Ditch radial resistance [d/m]    
    """
    dtypeGHB = np.dtype([('I', int), ('C', float), ('h', float)])
    
    # --- Het fdm grid
    xw, yw = xy 
    xmin, ymin, xmax, ymax = xw - Lmdl/2, yw - Lmdl/2, xw + Lmdl/2, yw + Lmdl/2
    nx, ny = int((xmax - xmin) / dxy) + 1, int((ymax - ymin) / dxy) + 1 
          
    gr = Grid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), [0.02, 0, -D])
    gr.crs = "EPSG:31370"
    
    # --- Indices of the well cell
    ixw, iyw= gr.ix(xw)[0]['idx'], gr.iy(yw)[0]['idx']

    # ---- Boundary array as in Modflow 5
    IBOUND = gr.const(1, dtype=int)

    # --- Model arrays
    kx, ky, kz = gr.const(k), gr.const(k), gr.const(k)
    L = 2 * b
    c = L * w
    
    # ---Translate c to the vertical conductance of the top layer
    kz[0] = 0.5 * gr.DZ[0] / c
    
    # --- Model boundaries
    HI = gr.const(0.)
    FQ = gr.const(0.)    
    FQ[-1, iyw, ixw] = Q
    
    # --- Water courses --> GHB boundary
    water_gdf = sw.clip_water_to_gr(gr)
    line_len_gdf = sw.line_length_per_gr_cell(gr, water_gdf) # Hass a column length
    
    Id = line_len_gdf.index[line_len_gdf['water_length_m'] > 0]
    GHB = np.zeros(len(Id), dtype=dtypeGHB)
    GHB['I'] = Id + gr.nx * gr.ny # --- Put GHB in second layer!
    GHB['C'] = line_len_gdf.loc[Id, 'water_length_m'] / w
    GHB['h'] = HI.ravel()[Id]
    
    # --- Simulate with water courses as GHB
    IBOUND[0] = 0
    out1 = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB)
    
    # --- Simulate with water courses replaced by aquitard on top
    IBOUND[0] = -1
    out2 = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)
    
    # -- Analytic De Glee
    lam = np.sqrt(k * D * c)
    r = np.sqrt((xw - gr.XM[0])**2 + (yw - gr.YM[0])**2)        
    phi_Glee = Q / (2 * np.pi * k * D) * K0(r/lam)

    # --- Show the results
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(
        f"Finite Diff Model: grid {L/1000}x{L/1000} km, cell_size={dxy} m, shape=(nz={gr.nz}, ny={gr.ny}, nx={gr.nx}))" + "\n" +
        fr"well: (Q={Q:.0f} m$^3$/d at (xw,yw)=({xw:.0f}, {yw:.0f})" +"\n" +
        fr"w={w:.1f} d/m, L={L:.0f} m, kD={k * D:.0f} m$^2$/d, c={c:.0f} d, $\lambda$={lam:.0f}m, rch=0 m/d",
        fontsize=12)
    ax.set_title("Stijghoogteverandering door onttrekking (sloten vs unforme weerstandslaag/De Glee)")

    # --- Get suitable contour levels
    phimin, phimax = np.floor(phi_Glee.min() * 10) / 10, np.ceil(phi_Glee.max() * 10) /10
    levels = np.arange(phimin, phimax, 0.05) 
    levels =np.unique(-np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

    # --- Contour for watercourses, replacement aquitard and De Glee with adapted lambda
    C1 = ax.contour(gr.xm, gr.ym, out1['Phi'][-1], colors='r', linewidths=1, levels=levels)
    C2 = ax.contour(gr.xm, gr.ym, out2['Phi'][-1], colors='k', linewidths=1, levels=levels)
    C3 = ax.contour(gr.xm, gr.ym, phi_Glee,        colors='gray', linewidths=0.75, levels=levels)
    
    # --- Label the contour lines
    ax.clabel(C1, levels=C1.levels, fmt='%.2f')
    ax.clabel(C2, levels=C2.levels, fmt='%.2f')
    ax.clabel(C3, levels=C3.levels, fmt='%.2f')
    
    # --- Plot the water courses
    water_gdf.plot(color='blue', linewidth=0.75, ax=ax)
    
    # --- Plot the well
    ax.plot(xw, yw, 'ro', ms=7, mec='k', mfc='r', label=f'well Q={Q} m3/d')
    
    # --- Plot the grid
    # gr.plot_grid(ax=ax, linewidth=0.5, color='g')

    ax.grid(which='both')
    ax.set_aspect(1)

    # --- Legend for each contour set
    ln1, = ax.plot([], [], 'r--')
    ln2, = ax.plot([], [], 'k--')
    ln3, = ax.plot([], [], '--', color='gray')
    
    ax.legend(handles=[ln1, ln2, ln3],
              labels=[f'GHB w={w:.1f} d/m',
                      rf"Fdm: $c=w L=${w:.1f}$\times${L:.0f}={c:.0f} d",
                      fr"De Glee: $\lambda=\sqrt{{kDc}}=\sqrt{{{k*D*c:.0f}}}$={lam:.0f} m"])
    
    logo(fig, os.path.basename(__file__))
    fig.savefig(os.path.join(images, f"fdm_de_Glee_xy_{xw:.0f}_{yw:.0f}.png"))
    

# --- Coordinaten ingreeplocaties
well_coordinates = (
    ((195000., 196500.), -3000, 600),
    ((193400., 198550.), -3000, 600),
    ((193500., 196200.), -3000, 800),   
    ((192300., 194100.), -3000, 600), 
    ((196250., 195600.), -3000, 350),   
    )


if __name__ == '__main__':
    L, D, k = 500, 25, 10
    w, xw = 0.2, 0
    ws=[0.25, 0.5, 1, 2]
    rDitch = 1
    if False:
        Qw = -1.0
        two_simultaneous_xsections(L=L, D=D, k=k, ws=ws, xw=xw, Qw=Qw, rDitch=rDitch)
    if True:
        Qw = -2400
        two_simultaneous_3Dmodels(L=L, D=D, k=k, ws=ws, xw=xw, Qw=Qw, rDitch=rDitch)
    if False:
        water_gdf = sw.clip_water_to_gr(gr)
        sw.distance_raster(water_gdf, pixelsize=10)

        dist_to_water = sw.distance_grid_with_grates(xy=(194000, 195000), L=15000, dxy=50)
        fig = plt.gcf()
        ax = fig.axes[0]
        for obs in well_coordinates:
            path = sw.climb_to_ridge(*obs, dist=dist_to_water)[[0, -1]]
            ax.plot(*path.T, 'r-')
            ax.plot(*path[0], 'o', ms=7, mec='r', mfc='white', alpha=1)
            ax.plot(*path[1], 'o', ms=7, mec='r', mfc='gold',  alpha=1)
        logo(fig, os.path.basename(__file__))
        fig.savefig(os.path.join(images, "afstand_tot_oppwater.png"))
    if False:
        for wel in well_coordinates:
            xy, Q, b = wel[0], wel[1], wel[2]
            Lmdl, D, k, w, dxy = 10000, 20, 10, 2.5, 50
            print(f"Running well Q={Q} at {xy[0]}{xy[1]}, b={b} m")
            fdm_real_surfwater(xy=xy, Q=Q, b=b, Lmdl=Lmdl, D=D, k=k, dxy=dxy, z=[0, -2, -2 - D], w=w)
        print("Done")

    plt.show()
    print("done")