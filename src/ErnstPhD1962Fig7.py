"""Study the breakup of the flow in fig7 in Ernst's PhD of 1962."""
# %%
# [tool.ruff]
# select = ["E", "F"]
# ignore = ["I001"]

import os
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from fdm.src import fdm3
from fdm.src.mfgrid import Grid


# %%
cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT") + 1], "Coding", "images")

# %% The freatic head with and without vertical

def xsection_7b():
    b, D, kh, kv, N = 100, 10, 5., 0.1, 0.001
    x = np.linspace(0, b, 1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Stroming in vert. sectie volgens Ernst (1952, fig 7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')

    x0s = np.linspace(0, b, 21)
    x0s[0] = 0.1

    for x0 in x0s:
        z0 = D
        z = x0 * z0 / x
        ax.plot(x[x>=x0], z[x>=x0])

    phi1 = D + N / (2 * kh * D) * (b ** 2 - x ** 2)
    phi2 = phi1 + N * D / (2 * kv) * (1 - (x/b) ** 2)

    ax.plot(x, phi1, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2, label="Freatisch vlak zonder verticale weerstand")
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(images, "Ernst_Xsec_basic"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Freatisch vlak met en zonder verticale weerstand naar Ernst (1952, fig 7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')
    ax.plot(x, phi1 - D, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2 - D, label="Freatisch vlak zonder verticale weerstand")

    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(images, "Ernst_Xsec_freatisch"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Freatisch vlak met en zonder verticale weerstand naar Ernst (1952, fig 7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')
    ax.plot(x, phi1 - D, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2 - D, label="Freatisch vlak zonder verticale weerstand")

    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(images, "Ernst_Xsec_freatisch"))

    plt.show()

def get_Z(b, D, N=100):
    x = np.linspace(0, b, int(b/D * N + 1))
    y = np.linspace(0, D, N+ 1)
    X, Y = np.meshgrid(x, y)
    Z = X.clip(1e-3, None) + 1j * (Y.clip(1e-3, D-1e-3))
    return Z



def stroming_analytisch(b=40, D=10, dxy=0.1, N=0.001, k=1, case=None, ax=None):    
    kh, kv = k, k
    Q = N * b
    dQ = Q / 20

    Z = get_Z(b, D)
    
    # Omega = N / (2 * D) * ((b**2 - Z.real**2) +  Z.imag**2  + 1j * Z.real * Z.imag)

    # --- Horizontal head loss (no contraction of stream lines)
    Phi1 = N / (2*D) * (b**2 - Z.real**2)
    
    # --- Vertical head loss (no contraction of stream lines)
    Phi2 = N * (D/2) * (Z.imag/D)**2
    
    # --- Psi = N/D * xy
    Psi = N/D  * Z.real * Z.imag
    
    # --- Total head loss (no contraction of stream lines)
    Phi = Phi1 + Phi2
    
    Omega0 = Phi - 1j * Psi

    phiLevels = np.arange(0, 20 * Q, dQ)
    psiLevels = np.arange(0, 1 * Q, dQ)

    # --- Extraction, extraction at point (0, iD)
    Omega1 = -Q/ (np.pi / 2) * np.log(np.sin(1j *np.pi/2 * ((b-Z)/D - 1j)))
    
    # --- Uniform flow to cancel the flow at x=b due to Omega1
    Omega2 =  +Q * ((b - Z)/D - 1j )


    if False:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Phi at different depths")
        for z, omega in zip(Z, Omega0):
            ax.plot(z.real, omega.real, label=f"x={z[0].imag}")


    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 11))
    fig.suptitle("Exacte anal. doorrek. vd samengestelde stroming volgens Ernst (1962, fig.7)"
                 "\n"
                 f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, Q=Nb={N * b} m2/d")

    titles = ["Combinatie van Ernst (1962), fig.7b en fig.7c",
              "Stroming door contractie van stroomlijnen (Ernst (1962), fig.7d)",
              "Samengestelde stroming in de doorsnede (Ernst (1962), fig.7a)"]
        
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set(xlabel='x[m]', ylabel='y[m]')
        if ax == axs[0]:
            Omega = Omega0
        elif ax == axs[1]:
            Omega = -(Omega1 + Omega2)
        elif ax == axs[2]:
            Omega = Omega0 - (Omega1 + Omega2)
        
        phimin = np.floor(Omega.real.min() * 100) / 100
        phimax = np.ceil(Omega.real.max() * 100) / 100
        psimin = np.floor(Omega.imag.min() * 100) / 100
        psimax = np.ceil(Omega.imag.max() * 100) / 100
        phiLevels = np.arange(phimin, phimax, dQ)
        psiLevels = np.arange(psimin, psimax, dQ)

        ax.contour(Z.real, Z.imag, Omega.real, levels=phiLevels,
                   colors='b',
                   linestyles='solid',
                   linewidths=0.5)
        
        ax.contour(Z.real, Z.imag, Omega.imag, levels=psiLevels,
                   colors='r',
                   linestyles='solid',
                   linewidths=0.5)
        ax.set_aspect(1)
        ax.set_ylim(0, 10)
        if not ax==axs[0]:
            ax.plot(b, D, 'ro', ms=20, mec='b', mfc='blue', zorder=100)

        fig.savefig(os.path.join(images, "Ernst_fig7_analytisch.png"))


def stroming_numeriek(b=20, D=10, dxy=0.1, N=0.01, k=1., case=5, ax=None):
    Q = N * b
    n = int(b / dxy) + 1
    m = int(D / dxy) + 1
    x = np.linspace(0, b, n+1)
    z = np.linspace(0, D, m+1)
    gr = Grid(x, None, z, axial=False)

    k = gr.const(k)
    FQ = gr.const(0.)
    HI = gr.const(0)

    IBOUND = gr.const(1, dtype=int)
    IBOUND[0, 0, 0] = -1

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if case in [0, 2, 4]:
        ax.set_ylabel('z[m]')
    if case in [3, 4]:
        ax.set(xlabel='x[m]')

    phiLevels = 20
    psiLevels = 20

    if case==0:
        ax.set_title("Ernst fig 7a")
        FQ[0, 0, : ] = N * gr.dx
    elif case==1:
        ax.set_title("Ernst fig 7b")
        FQ = -N/D * gr.DX * gr.DZ
        FQ[0, 0, :] += N * gr.dx        
    elif case==2:
        ax.set_title("Ernst fig 7c")
        FQ = +N/D * gr.DX * gr.DZ
        FQ[:, 0, 0] -= N *b / D * gr.dz        
    elif case==3:
        ax.set_title("Ernst fig 7b + 7c")
        FQ = -N/D * gr.DX * gr.DZ
        FQ[0, 0, :] += N * gr.dx    
        FQ += N/D * gr.DX * gr.DZ
        FQ[:, 0, 0] -= N *b / D * gr.dz
    elif case==4:
        ax.set_title("Ernst fig 7d")
        FQ[:, 0, 0] = N * b / D * gr.dz
    elif case==5:
        ax.set_title("Ernst fig 7a")
        FQ[0, 0, : ] = N * gr.dx
        
    ax.plot(gr.xm[0], gr.zm[0], 'ro', ms=20, mec='b', mfc='blue', zorder=100)
    
    out = fdm3.fdm3(gr, K=k, c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)
    out['gr'] = gr
    if case not in [1, 2]:
        dQ = Q / 20
        print(f"phiMn={out['Phi'].min():.4g}, phiMax={out['Phi'].max():.4g}")
        phiLevels = np.arange(out['Phi'].min(), out['Phi'].max(), dQ)
            
        sf = fdm3.psi(out['Qx'])
        print(f"psiMin={sf.min():4g}, psiMax={sf.max():.4g}")        
        psiLevels = np.arange(sf.min(), sf.max(), dQ)
        
    Cf = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], out['Phi'][:, 0, :],  levels=phiLevels,
                    colors='b',
                    linewidths=0.5,
                    linestyles='solid')
    # ax.clabel(Cf, levels=Cf.levels)
    
    if case not in [1, 2]:
        Cs = ax.contour(gr.X[:, 0, 1:-1], gr.Z[:, 0, 1:], sf, levels=psiLevels,
                        colors='r',
                        linewidths=0.5,
                        linestyles='solid')
        # ax.clabel(Cs, levels=Cs.levels)
    
    ax.set_aspect(1)
    # ax.grid()
    ax.legend()
    print("Done")
    return out



# %% Complexe stroming in hoek
if __name__ == '__main__':
    
    b, D, N, k = 20, 10, 0.001, 1
    
    if False:
        stroming_analytisch(b=b, D=D, dxy=0.1, N=N, k=k, case=None, ax=None)
        
    if True: 
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.suptitle("Narekenen Ernst (1962, fig 7)\n"
                      f"b={b} m, D={D} m, k={k} m/d, N={N} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k)} [d]")
        ax2.set_title("Stijghoogte aan freatisch vlak in de verschillende deelstroomfiguren")
        ax2.set(xlabel='x[m]', ylabel=r'$\phi - \phi_0$ [m]')
        ax2.grid(True)
           
        fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(12, 10))
        fig.suptitle("Ernst (1962) Fig 7, numeriek. (Phi=blauw, Psi=Rood)\n"
                     f"b={b} m, D={D} m, N={N} m/d, k={1} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k)} [d]")
        
        clrs = cycle('brgkmc')
        for ic, ax in enumerate(axs.flatten()):
            out = stroming_numeriek(b=b, D=D, dxy=0.1, N=N, k=k, case=ic, ax=ax)
            fig.savefig(os.path.join(images, "ErnstFig7_numeriek.png"))

            if ic > 0:
                clr = next(clrs)
                if ic + 1 == 3:
                    ax2.plot(out['gr'].xm[::5], out['Phi'][0, 0, ::5], '.-', color=clr, label=f"freatisch deelfiguur {ic + 1}")
                    ax2.plot(out['gr'].xm[::5], out['Phi'][-1, 0, ::5], '.--', color=clr, label=f"basis, deelfiguur {ic + 1}")
                else:
                    ax2.plot(out['gr'].xm, out['Phi'][0, 0, :], ls='solid', color=clr, label=f"freatisch deelfiguur {ic + 1}")
                    ax2.plot(out['gr'].xm, out['Phi'][-1, 0, :], ls='dashed', color=clr, label=f"basis, deelfiguur {ic + 1}")
    ax2.legend(loc='center')
    fig2.savefig(os.path.join(images, "ErnstNumeriekPhimaaiveld.png"))

    plt.show()
    print("done")
