# %% [markdown]
## Voortoets functionality
# 
#In de voortoets worden verschillende formules gebruikt voor het berekenen van het # effect van ingrepen in de waterhuishouding op het grondwater. De implementatie van de verschillende formules in Python is naar dit bestand verhuisd om de Python code in het hoofdbestand zo kort en krachtig mogelijk te houden.
# 
# @TO 2025--02-26
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from etc import newfig, attr
from scipy.signal import lfilter, filtfilt
from scipy.special import k0 as K0
from scipy.integrate import quad, quad_vec
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# %%
class Dirs:
    """Namespace for directories in project"""
    def __init__(self):
        self.home = '/Users/Theo/Entiteiten/Hygea/2022-AGT/jupyter/'
        self.data = os.path.join(self.home, 'data')
        self.images = os.path.join(self.home, 'images')
    
dirs = Dirs()
os.path.isdir(dirs.data)
os.path.isdir(dirs.images)

# %%
class Mirror_ditches():
    """Class to compute mirror ditches plus signs or mirror wells plus signs
    
    In case of ditches the position of the ditches are returned with their signs.
    In case of a well, the mirror well positions are returned.
    """
    def __init__(self, xL, xR, xw=None, N=30, Lclosed=False, Rclosed=False):
        """Return x-coordinate of mirror points given a strip between xL and xR, xp of point of well.

        Parameters
        ----------
        xL, xR: floats
            x-coordinates of left and right of strip between the two head boundaries
        N: int
            number of mirror wells positions
        """
        if xL > xR:
            xL, xR = xR, xL
        self.xw = xw
        self.xL = xL
        self.xR = xR
        self.Lclosed = Lclosed
        self.Rclosed = Rclosed
        self.N = N
        
        # The actual well (May be None for just mirroring ditches)
        self.xw = xw
        self.sw = 1

        # Starting values for the coordinates of the mirrors
        # The right ditch (xR) and the left ditch (xL)
        xRD, xLD = [xR], [xL]
        
        # first mirror wells signs are opposite to self.sw
        if self.xw is not None:
            sRD = [self.sw if Rclosed else -self.sw]
            sLD = [self.sw if Lclosed else -self.sw] 
        
        # No wells, just ditches. First ditches have positive sign
        # that may be inverted later based on Lclosed and Rclosed
        else:
            sRD, sLD = [1], [1]

        # Get mirror ditch or well coordinates starting with
        # either the right ditch (xRD) or with the left  ditch (xLD)
        for i in range(1, N):
            if i % 2 == 1:
                xRD.append(xL - (xRD[-1] - xL))
                xLD.append(xR - (xLD[-1] - xR))
                sRD.append(sRD[-1] if Lclosed else -sRD[-1])
                sLD.append(sLD[-1] if Rclosed else -sLD[-1])
            else:
                xRD.append(xR - (xRD[-1] - xR))
                xLD.append(xL - (xLD[-1] - xL))
                sRD.append(sRD[-1] if Rclosed else -sRD[-1])
                sLD.append(sLD[-1] if Lclosed else -sLD[-1])
                
        self.xLD = xLD
        self.xRD = xRD
        self.sLD = sLD
        self.sRD = sRD
        
        # If a well position was given we don't want the mirror ditches but the mirror wells:
        if self.xw is not None:
            xM = 0.5 * (xR + xL)
            deltaR = xR - xw     # mirror position over right ditch
            deltaL = xL - xw     # mirror position over left  ditch
            for i in range(len(xRD)):
                if self.xRD[i] > xM:
                    self.xRD[i] += deltaR
                else:
                    self.xRD[i] -= deltaR
            for i in range(len(xLD)):
                if self.xLD[i] < xM:
                    self.xLD[i] += deltaL
                else:
                    self.xLD[i] -= deltaL
        return

    def show(self, ax=None, figsize=(8, 2), fcs=('yellow', 'orange')):
        """Return picture of the ditches and the direction of the head change.
        
        Parameters
        ----------
        ax: matplotlib.Axes.axes or None
            axis to plot on
        fcs: tuple  of two strings
            colors of the mirror strips and the central strips respectively
        """
        L = self.xR - self.xL
        
        if not ax:
            _, ax = plt.subplots(figsize=figsize)   

        # Draw arrows for xLD ditches
        for x, sgn in zip(self.xLD, self.sLD):
            if sgn > 0:
                y1, y2 = sgn, 0   # upward arrow
            else:
                y1, y2 = 0, -sgn  # downward arrow
            ax.annotate("", xy=(x, y1), xytext=(x, y2), arrowprops=dict(arrowstyle="->", color="red"))
            
        # Draw arrow for xRD ditches
        for x, sgn in zip(self.xRD, self.sRD):
            if sgn > 0:
                y1, y2 = sgn , 0
            else:
                y1, y2 = 0, -sgn
            ax.annotate("", xy=(x, y1), xytext=(x, y2), arrowprops=dict(arrowstyle="->", color="blue")) 

        # Plot the strips and accentuate the central one
        for xl in np.sort(np.hstack((self.xL - np.arange(self.N) * L, self.xR + np.arange(self.N - 1) * L))):
            xr = xl + L
            fc = fcs[1] if xl < 0 and xr > 0 else fcs[0]        
            p = Path(np.array([[xl, xr, xr, xl, xl], [0, 0, -1, -1, 0]]).T, closed=True)
            ax.add_patch(PathPatch(p, fc=fc, ec='black'))

        ax.plot(0.5 * (self.xL + self.xR), 0, 'ro')
        
        if self.xw is not None:
            ax.annotate("", xy=(self.xw, self.sw), xytext=(self.xw, 0), arrowprops=dict(arrowstyle="->", color="green")) 
        
        ax.set_title(f"Mirror ditches with     Left: {"closed" if self.Lclosed else "open"}, Right: {"closed" if self.Rclosed else "open"}")
        ax.set_ylim(-1, 1.1)
        
        # Don't want no yticks and no ticklabels
        ax.set_yticks([])
        return ax

if __name__ == "__main__":
    
    # Example, Use of mirror ditches, which immediately shows all 4 possibilities
    # The problem is an aquifer between xL and xR with as boundaries that the aquifer
    # at xL and or xR is fixed or closed.
    
    kD, S, c= 600, 0.001, 200
    lambda_ = np.sqrt(kD * c)
    
    xL, xR, xw, N= -200, 200, 0, 30
    Lclosed, Rclosed = False, False # right and or left size of strip closed?
    L = xR - xL
    n = 0
    x = np.linspace(xL - n * L, xR + n * L, 1201)
    Q = -1200.

    fig, ax = plt.subplots()
    md = Mirror_ditches(xL, xR, xw=xw, N=N, Lclosed=Lclosed, Rclosed=Rclosed)
    ax = md.show(ax=ax, figsize=(8, 2), fcs=('yellow', 'orange'))

    s = np.zeros_like(x)
    r = np.sqrt((md.xw - x) ** 2)
    s = md.sw * Q / (2 * np.pi * kD) * K0( r / lambda_)
    for xw, sgn in zip(md.xLD, md.sLD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)
    for xw, sgn in zip(md.xRD, md.sRD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)
        
    _, ax = plt.subplots()
    ax.set_title("Result of flow caused by a well between fixed boundaries")
    ax.set(xlabel="x [m]", ylabel='head [m]')
    ax.plot(x, s)
    ax.grid()
    plt.show()