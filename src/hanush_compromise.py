# Hantush als compromis voor de berekeing van de verlaging van freatische grondwater

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fdm.src.fdm3 import fdm3
from fdm.src.mfgrid import Grid

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

def two_simlataneous_xsections(L=500, D=25, k=1, w=1, xw=2500, Qw=None, rDitch=1):
       
    a =0.01

    # --- The grid
    x = sinspace(x1=0, x2=500, th1=np.pi *a, th2=np.pi * (1-a), N=25)
    x = concat_space(x, 10)
    z = np.array([0, -2, -2-D])

    # --- Two rows, first ditches, second semi-confined
    gr = Grid(x, y=np.array([0, 1, 2]), z=z)

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

    # --- Radiale weerstand [d/m]
    if w is None:
       w = 1 / (np.pi * k) * np.log(D / (np.pi * rDitch))
    
    # --- Weerstand tweede doorsnde compatibel met die van de sloten [d]
    c = w * L

    kx[0] = 1e-20
    # --- Begin kz[0] nul te maken
    kz[0, 0, :] = 1e-20
    # --- Dan in de doorsnede met de sloten:
    kz[0, 0, od] = gr.DZ[0, 0, od] / (2 * w * gr.DX[0, 0, od])
    kz[1, 0, od] = 1000.
    # --- Voorts in de doorsnede met de scheidende laag
    kz[0, 1, :] = gr.DZ[0, 1, :] / (2 * c)

    HI = gr.const(0.)
    FQ = gr.const(0.)

    # --- De onttrekking in een keer in beide doorsnedes
    FQ[1, :, iw] = Qw # --- Both rows (models)

    # --- Boundary array
    IBOUND = gr.const(1, dtype=int)
    # --- Eerste doorsnede zet cellen met sloten fixed head.
    IBOUND[0, 0, od] = -1
    # --- Tweede doorsnede zet alle cellen in de toplaag fixed head
    IBOUND[0, 1, :] = -1

    # ---Simulate
    out = fdm3(gr, K=(kx, ky, kz), c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)

    # --- Analytisch
    # --- Doorsnede 1 analytisch gebaseerd op out['Qx']
    # ----We nemen Qx en berekenen s[i+1] = s[i] - Qw/kD dx
    kD = k*D
    Qx = out['Qx']
    s1 = np.zeros_like(gr.x[1:-1])
    for i in range(1, len(s1)):
        s1[i] = s1[i-1] - Qx[-1,0,i] * gr.dx[i+1] / kD        

    # --- Doorsnede 2 analytisch
    lam = np.sqrt(kD*c)
    s2 = lam/kD * Qw/2 * np.exp(-np.abs(gr.xm-xw)/lam)

    # --- Show
    fig, ax = plt.subplots(figsize=(10, 8))
    title=(f"Extraction Qw={Qw} m2/d at x={gr.xm[iw]:.0f}" + f", kD={k * D} m2/d" + "\n" +
        f"w={w:.3f} d/m, c={c:.3f} d, lambda={lam:.0f}"
    )
    ax.set(title=title, xlabel="x[m]", ylabel="head[m]")

    for iL in range(1, gr.nz):
        ax.plot(gr.xm, out['Phi'][iL, 0, :], label=f'layer[{iL}], with ditches')
        ax.plot(gr.xm, out['Phi'][iL, 1, :], label=f'layer[{iL}], semi-confined')
        
    ax.plot(gr.x[1:-1], s1, '+', ms=2, label='met sloten, analytisch')    
    ax.plot(gr.xm,      s2, 'x', ms=2, label='met scheidende laag, analytisch')
        
    ax.plot(gr.xm[od],gr.xm[od] * 0, 'v', ms=10, mec='r', mfc='none', label='ditch locations')

    ax.grid()
    ax.legend()    
    

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
    w, xw, Qw = 1, 2500, -1
    rDitch = 1
    if True:
        two_simlataneous_xsections(L=L, D=D, k=k, w=w, xw=xw, Qw=Qw, rDitch=rDitch)
    if False:
        analytic_test(L=500, D=25, k=1, w=1, xw=xw, Qw=Qw, rDitch=1)

    plt.show()
    print("done")