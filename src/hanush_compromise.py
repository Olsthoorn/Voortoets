# Hantush als compromis voor de berekeing van de verlaging van freatische grondwater

# %%
import os
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

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

    # --- Two rows, first ditches, second semi-confined
    gr = Grid(x - x.mean(), y=np.array([0, 1, 2]), z=z)

    # --- Where are the ditches?
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
        "Blue: recharge via ditches. Red: recharge via resistance top layer.\n"
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
        # --- In Xsec with ditches, computed adapt kz in both layers to match the given resistance
        Dx = gr.dx[od]
        w2 = gr.DZ[1, 0, od] / (2*Dx * kz[1, 0, od])  # w of second layer
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
        # --- Xsec 1, with ditches
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
            ax.plot(gr.xm, out['Phi'][iL, 0, :], color=clr, label=f'layer[{iL}], with ditches')
            clr = next(clrs)
            ax.plot(gr.xm, out['Phi'][iL, 1, :], color=clr, label=f'layer[{iL}], semi-confined')
            
        clr = next(clrs)
        ax.plot(gr.xm[od], phiX1[od], 's', ms=4, mfc='none', mec=clr, label='With ditches, analytic')  
        ax.plot(gr.xm[:], s2[:],  '.', mfc='none', mec=clr, label='With separating layer anlaytic')
            
        ax.plot(gr.xm[od],gr.xm[od] * 0, 'v', ms=6, mec='b', mfc='none', label='ditch locations')
        xl = np.ceil((3 * lam) / 100) * 100
        ax.set_xlim(-xl, xl)

        ax.grid()
        ax.legend(loc='lower left')
    fig.savefig(os.path.join(images, "drawdown_xsec_4xw.png"))
    
    print(fr"lam/L = {np.sqrt(kD * np.array(ws) / L)}")
        

def analytic_test(L=500, D=25, k=1, w=0.3, xw=2500, Qw=None, rDitch=1):
       
    a =0.01

    # --- The grid
    x = np.linspace(0, 5000, 500)
    z = np.array([0, -2, -2-D])

    # --- Two rows, first ditches, second semi-confined
    gr = Grid(x, y=None, z=z)

    # --- The well
    iw = gr.ix(xw)['idx'][0]
    iw=0
    
    # --- Model arrays
    kx, ky, kz = gr.const(k), gr.const(k), gr.const(k)
    

    # --- Weerstand tweede doorsnde compatibel met die van de sloten [d]
    c = w * L

    kz[0, :, :] = gr.DZ[0, :, :] / (2 * c)
    
    HI = gr.const(0.)
    FQ = gr.const(0.)

    # --- De onttrekking in een keer in beide doorsnedes
    FQ[1, :, iw] = Qw # --- Both rows (models)

    # --- Boundary array
    IBOUND = gr.const(1, dtype=int)
    IBOUND[0] = -1
    
    # ---Simulate
    out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)

    # --- Analytisch
    kD = k*D
    lam = np.sqrt(kD*c)
    s2_ = lam/kD * Qw/2 * np.exp(-np.abs(gr.xm-gr.xm[iw])/lam)

    # --- Show
    fig, ax = plt.subplots(figsize=(10, 8))
    title=(f"Extraction Qw={Qw} m2/d at x={gr.xm[iw]:.0f}" + f", kD={k * D} m2/d" + "\n" +
        f"w={w:.3f} d/m, c={c:.3f} d"
    )
    ax.set(title=title, xlabel="x[m]", ylabel="head[m]")

    for iL in range(1, gr.nz):
        ax.plot(gr.xm, out['Phi'][iL, 0, :], label=f'layer[{iL}], semi-confined')

    ax.plot(gr.xm,      s2_, 'o', ms=4, label='met scheidende laag, analytisch')
    ax.grid()
    ax.legend()    
    
  
    
if __name__ == '__main__':
    L, D, k = 500, 25, 10
    w, xw, Qw = 0.2, 0, -1
    ws=[0.25, 0.5, 1, 2]
    rDitch = 1
    if True:
        two_simlataneous_xsections(L=L, D=D, k=k, ws=ws, xw=xw, Qw=Qw, rDitch=rDitch)
    if False:
        analytic_test(L=500, D=25, k=1, w=1, xw=xw, Qw=Qw, rDitch=1)

    plt.show()
    print("done")