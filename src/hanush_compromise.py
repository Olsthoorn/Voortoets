# Hantush als compromis voor de berekeing van de verlaging van freatische grondwater

# %%
import os
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import k0 as K0
from tools.fdm.src.fdm3 import fdm3
from tools.fdm.src.mfgrid import Grid

import vtl_surf_water as sw

cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT-Voortoets") + 1], "Coding", "images")

# %%
def sinspace(x1, x2, th1=0, th2=np.pi, N=100):
    ds = np.sin(np.linspace(th1, th2, N))
    x = x1 + (x2 - x1) * np.hstack((0, np.cumsum(ds) / ds.sum()))
    return x

def concat_space(x, N):
    x_ = x
    for i in range(N-1):
        x = np.hstack((x[:-1], x[-1] + x_))
    return x

def two_simlataneous_xsections(L=500, D=25, k=1, ws=[0.25, 0.5, 1, 2], xw=2500, Qw=None, rDitch=1):
       
    a =0.01

    # --- The grid
    x = sinspace(x1=0, x2=500, th1=np.pi *a, th2=np.pi * (1-a), N=25)
    x = concat_space(x, 20)
    z = np.array([0, -2, -2-D])

    # --- Two rows, second ditch_mdl, ditch_mdl semi-confined
    gr = Grid(x - x.mean(), y=np.array([0, 1, 2]), z=z)

    # --- Where are the ditch_mdl?
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
        fontsize=16)

    for w, ax in zip(ws, axs.ravel()):
        clrs = cycle('brgkmc')
        # --- Radiale weerstand [d/m]
        if w is None:
            w = 1 / (np.pi * k) * np.log(D / (np.pi * rDitch))
        
        # --- Weerstand tweede doorsnde compatibel met die van de sloten [d]
        c = w * L

        kx[0] = 1e-20        
        kz[0, 0, :] = 1e-20 # --- Start making kz 0
        # --- In Xsec with ditch_mdl, computed adapt kz in both layers to match the given resistance
        Dx = gr.dx[od]
        w2 = gr.DZ[1, 0, od] / (2*Dx * kz[1, 0, od])  # w of ditch_mdl layer
        w1 = w - w2                                   # w of top layer
        w1[np.isclose(w1,0)] = 1e-6                   # prevend division by zero
        kz[0, 0, od] = gr.DZ[0, 0, od] / (2*Dx * w1)  # matching kz in top layer
        
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
        out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)

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
    fig.savefig(os.path.join(images, "drawdown_xsec_4xw.png"))
    
    print(fr"lam/L = {np.sqrt(kD * np.array(ws) / L)}")
        
def two_simlataneous_3Dmdoels(L=500, dxy=25, k=10, D=20, ws=[0.25, 0.5, 1, 2], xw=2500, Qw=-2400, rDitch=1):
    """Generate and run two jumeleaux (back-to-back) 3D (two-layer) FDM models.
    
    There is one well. This allows simulating only one halfspace in each model.
    The second model has parallel ditch_mdl as boundaries L apart.
    The ditch_mdl model has an equivalent resistance layer on top.
    The models are separated from each other by a row in the commenter
    between y=-0.001 and y=+0.001 with zero conductivity.    
    """
       
    a =0.01

    # --- The grid
    x = sinspace(x1=0, x2=500, th1=np.pi *a, th2=np.pi * (1-a), N=25)
    x = concat_space(x, 20)
    z = np.array([0, -2, -2-D])
    y= np.logspace(0, np.log10(5000), 51)
    y = np.hstack((-y[::-1], -0.001, 0.001, y))
    
    # --- Two rows, second ditch_mdl, ditch_mdl semi-confined
    gr = Grid(x - x.mean(), y=y, z=z)
    
    # Masks to select parts of the model
    ditch_mdl = gr.ym > 0 # Model with the ditch_mdl
    clayer_mdl  = gr.ym < 0 # Model with the resitance top layer
    wall = np.logical_and(np.logical_not(clayer_mdl), np.logical_not(ditch_mdl)) # The row in between the two models

    # --- Where (what ix) are the ditches in the ditch_mdl?
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
    
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(10, 11))
    fig.suptitle(
        f"{gr.dx.sum():.0f} m wide and long 3D twin models each with a single well\n"
        "Blue: recharge via ditches. Red: recharge via resistance top layer.\n"
        fr"$Q_w$={Qw} m3/d at $(x,y)$=({gr.xm[iw]:.0f},0)" + f", $kD$={k * D} m2/d",
        fontsize=14)

    for w, ax in zip(ws, axs.ravel()): # w is ditch resistance [d/m]        
        # --- Radial resistance [d/m]
        if w is None:
            w = 1 / (np.pi * k) * np.log(D / (np.pi * rDitch))
        
        # --- Equivalent resistance layer compatible with the ditch_mdl [d]
        c = w * L

        kx[0] = 1e-20 # --- Start makking kx[0] zero
        kz[0] = 1e-20 # --- Start making kz[0] zero
        # --- In model with ditch_mdl, adapt kz to match the given ditch resistance
        ditch_od = np.ix_(ditch_mdl, od)        
        w2 = gr.DZ[1][ditch_od] / (2* gr.DX[1][ditch_od] * kz[1][ditch_od])  # w of ditch_mdl layer [d/m2]
        w1 = w - w2                                   # w of top layer
        w1[np.isclose(w1,0)] = 1e-6                   # prevent division by zero
        kz[0][ditch_od] = gr.DZ[0][ditch_od] / (2*gr.DX[0][ditch_od] * w1)  # matching kz in top layer
        
        # --- Set kz in Xsec with resistance layer
        kz[0, clayer_mdl, :] = gr.DZ[0, clayer_mdl, :] / (2 * c)

        HI = gr.const(0.)
        FQ = gr.const(0.)

        FQ[1, iyw, iw] = Qw / 2 # --- Both rows (models) get the same well

        # --- Boundary array
        IBOUND = gr.const(1, dtype=int)        
        IBOUND[0][ditch_od] = -1 # Model 1: with cells having fixed head.
        IBOUND[0, clayer_mdl,  :] = -1 # Model 2: with equivalent resistance layer.
        IBOUND[:, wall, :] = 0
        
        # ---Simulate
        out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)

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
    fig.savefig(os.path.join(images, "drawdown_3D_4xw.png"))
    
    print(fr"lam/L = {np.sqrt(kD * np.array(ws) / L)}")


def fdm_real_surfwater(xy=None, Q=-3000, b=500, L=15000, D=20, k=10, dxy=50, z=[0, -2, -22], w=2):
    """Generate and run a LxL sized model with extraction Q, in location xy. 
    
    The model has two layers one to host the wells and and for the regional aquifer.
    The radial surface water resistance is w [d/m]
    The surface water is obtained from the OSM of Belgium.
    """
    dtypeGHB = np.dtype([('I', int), ('C', float), ('h', float)])
    
    xw, yw = xy 
    xmin, ymin, xmax, ymax = xw - L/2, yw - L/2, xw + L/2, yw + L/2
    nx, ny = int((xmax - xmin) / dxy) + 1, int((ymax - ymin) / dxy) + 1 
          
    gr = Grid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), [0.02, 0, -D])
    gr.crs = "EPSG:31370"
    
    # --- Indices of well cell
    ixw, iyw= gr.ix(xw)[0]['idx'], gr.iy(yw)[0]['idx']

    IBOUND = gr.const(1, dtype=int)

    # --- Model arrays
    kx, ky, kz = gr.const(k), gr.const(k), gr.const(k)
    c = 2 * b * w
    
    # --- make lambda 2b (trial)
    lamb = 2 * b
    c = lamb ** 2 / (k * D)

    kz[0] = 0.5 * gr.DZ[0] / c
    lam = np.sqrt(k * D * c)
    
    HI = gr.const(0.)
    FQ = gr.const(0.)    
    FQ[-1, iyw, ixw] = Q
    
    water_gdf = sw.clip_water_to_gr(gr)
    line_len_gdf = sw.line_length_per_gr_cell(gr, water_gdf) # Hass a column length
    
    Id = line_len_gdf.index[line_len_gdf['water_length_m'] > 0]
    GHB = np.zeros(len(Id), dtype=dtypeGHB)
    GHB['I'] = Id + gr.nx * gr.ny
    GHB['C'] = line_len_gdf.loc[Id, 'water_length_m'] / w
    GHB['h'] = HI.ravel()[Id]
    
    out1 = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB)
    
    IBOUND[0] = -1
    out2 = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)
    
    r = np.sqrt((xw - gr.XM[0])**2 + (yw - gr.YM[0])**2)
    phi_Glee = Q / (2 * np.pi * k * D) * K0(r/lam)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(
        f"Finite Diff Model: grid {L/1000}x{L/1000} km, cell_size={dxy} m, shape=(nz={gr.nz}, ny={gr.ny}, nx={gr.nx}))\n"
        f"well: (Q={Q:.0f} m3/d at (xw,yw)=({xw:.0f}, {yw:.0f}), kD={k * D:.0f} m2/d\n" +
        f"w={w:.1f} d/m, b={b:.0f} m, c={c:.0f} d, lambda={lam:.0f} m, recharge=0",
        fontsize=12)
    ax.set_title("Stijghoogteverandering door onttrekking (sloten vs unforme weerstandslaag/De Glee)")

    phimin, phimax = np.floor(phi_Glee.min() * 10) / 10, np.ceil(phi_Glee.max() * 10) /10
    levels = np.arange(phimin, phimax, 0.05) 
    levels =np.unique(-np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

    C1 = ax.contour(gr.xm, gr.ym, out1['Phi'][-1], colors='r', linewidths=1, levels=levels)
    C2 = ax.contour(gr.xm, gr.ym, out2['Phi'][-1], colors='k', linewidths=1, levels=levels)
    # C3 = ax.contour(gr.xm, gr.ym, phi_Glee,        colors='k', linewidths=0.75, levels=levels)
    
    ax.clabel(C1, levels=C1.levels, fmt='%.2f')
    ax.clabel(C2, levels=C2.levels, fmt='%.2f')
    # ax.clabel(C3, levels=C3.levels, fmt='%.2f')
    
    water_gdf.plot(color='blue', linewidth=0.75, ax=ax)
    ax.plot(xw, yw, 'ro', ms=7, mec='k', mfc='r', label=f'well Q={Q} m3/d')
    
    # gr.plot_grid(ax=ax, linewidth=0.5, color='g')

    ax.grid(which='both')
    ax.set_aspect(1)
    
    fig.savefig(os.path.join(images, f"fdm_de_Glee_xy_{xw:.0f}_{yw:.0f}_2b.png"))
    
    
    

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
    w, xw, Qw = 0.2, 0, -1
    ws=[0.25, 0.5, 1, 2]
    rDitch = 1
    if False:
        two_simlataneous_xsections(L=L, D=D, k=k, ws=ws, xw=xw, Qw=2400, rDitch=rDitch)
    if False:
        two_simlataneous_3Dmdoels(L=L, D=D, k=k, ws=ws, xw=xw, Qw=-2400, rDitch=1)
    if False:
        analytic_test(L=500, D=25, k=1, w=1, xw=xw, Qw=Qw, rDitch=1)
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
        fig.savefig(os.path.join(images, "afstand_tot_oppwater.png"))
    if False:
        for wel in well_coordinates:
            xy, Q, b = wel[0], wel[1], wel[2]
            L, D, k, w, dxy = 10000, 20, 10, 2, 50
            print(f"Running well Q={Q} at {xy[0]}{xy[1]}, b={b} m")
            fdm_real_surfwater(xy=xy, Q=Q, b=b, L=L, D=D, k=k, dxy=dxy, z=[0, -2, -2 - D], w=w)
        print("Done")
    if True:
        fname = "Artikel-sloten_vs_claag_20260130_TO.tex"
        with open(os.path.join(images, '../docs/Rapportage', fname), "rb") as f:    
            for i, line in enumerate(f, 1):
                if b'\x05' in line:
                    print(f"Line {i} has ASCII 5 (^E)")

    plt.show()
    print("done")