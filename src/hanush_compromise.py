# Hantush als compromis voor de berekeing van de verlaging van freatische grondwater

# %%
import os
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import k0 as K0

from fdm.src.fdm3 import fdm3
from fdm.src.mfgrid import Grid

cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT") + 1], "Coding", "images")

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
        

def two_simlataneous_3Dmdoels(L=500, D=25, k=1, ws=[0.25, 0.5, 1, 2], xw=2500, Qw=-2400, rDitch=1):
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


if __name__ == '__main__':
    L, D, k = 500, 25, 10
    w, xw, Qw = 0.2, 0, -1
    ws=[0.25, 0.5, 1, 2]
    rDitch = 1
    if False:
        two_simlataneous_xsections(L=L, D=D, k=k, ws=ws, xw=xw, Qw=2400, rDitch=rDitch)
    if True:
        two_simlataneous_3Dmdoels(L=L, D=D, k=k, ws=ws, xw=xw, Qw=-2400, rDitch=1)
    if False:
        analytic_test(L=500, D=25, k=1, w=1, xw=xw, Qw=Qw, rDitch=1)

    plt.show()
    print("done")