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
from scipy.special import k0 as K0, erfc
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
class Mirrors():
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
    
# %% Generate a line that can illustrate ground surface elevation

def ground_surface(x, xL=None, xR=None, yM=1, Lfilt=20, seed=3):
    """Return a ground surface elevation for visualization purposes.
    
    The elevation will be 0 at xL and xR and approach yM in the middle.
    
    Random numbers are used that are smoothed by filfilt using integer Lfilt
    as the filter width, and are finally multiplied
    
    $$ y_M \times np.sqrt{\cos \left(\pi \frac{x - xM}{L}\right)}$$
    
    where $yM$ is the maximum elevation.

    Lfilt can be adapted to the number points in x.

    Parameters
    ----------
    x : ndarray
        Coordinates.
    xL, xR : floats, optional
        Left and right coordinate bounds for the surface. Defaults to first and last x values.
    yM : float, optional
        Approximate maximum height of the surface before smoothing. Default is 1.
    Lfilt : int, optional
        Length of the smoothing filter applied with scipy.signal.filtfilt. Default is 20.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Smooth ground surface elevation.
    
    @TO 2025-02-10
    """
    # The points of zero elevation, xL and xR are optional
    if xL is None:
        xL = x[0]
    if xR is None:
        xR = x[-1]
        
    L, xM = xR - xL, 0.5 * (xL + xR)
    
    y = yM * np.sqrt(np.abs(np.cos(np.pi * (x - xM) / L)))

    # Ensure Lfilt is at least 2 to avoid issues with filtfilt
    Lfilt = max(Lfilt, 2)

    # Random noise generation with optional seed
    rng = np.random.default_rng(seed)
    z = rng.random(len(x))

    # Apply smoothing and return elevation (method 'gust' prevents padding problems)
    filtered = y * filtfilt(np.ones(Lfilt) / Lfilt, 1, z, method='gust')
    return filtered

# %%

def Q_hs_1d(x, N=None, kD=None, xL=None, xR=None, hL=None, hR=None, lamb_L=None, lamb_R=None):
    """Return flow (discharge) in unconfined aquifer with leaky adjacent aquifers
    
    The aquifer has constant kD throughout. For xL < x< xR it is recharged by
    recharge N and for x < xL and x > xR the aquifer is leaky with the
    given values for lamb_L and lamb_R. Then both lamb_L and lamb_R are zero, 
    then the head will be fixed at hL and hR respectively at xL and xR.
    This head, therefore shows the head in a strip of land between xL and xR
    bounded on both sides to marshy or well-drained areas characterized by
    their characteristic length lambda, which can be different at the left and
    at the right.
    
    Parameters
    ----------
    x: np.ndarray or float
        coordinate where head is computed
    kD: float
        transmissivity
    xL, xR: floats
        Area boundary where elevation can be set.
    hL, hR: floats
        Left and right head, achieved when lambdas are zero.
    lamb_L, lamb_R: floats
        Characteristic length of left and right area.
    eps: float
        small tolerance.
    """
    if lamb_L is None:
        lamb_L = 0.
    if lamb_R is None:
        lamb_R = 0.
        
    # Discharge at xL
    L = xR - xL
    QL = (-kD / (L + lamb_L + lamb_R)  * (hR - hL + N / (2 * kD) * L ** 2 +
                                          N / kD * L * lamb_R))
    QR = QL + N * L
    
    print(f"QL = {QL:.4g}, QR = {QR:.4g}")

    x = np.atleast_1d(x).astype(float)
    Q = np.zeros_like(x)
    
    # Deal with the three area separately
    mask_L = x < xL
    mask_R = x > xR
    mask_C = ~(mask_L | mask_R)

    Q[mask_L] = QL * np.exp(+(x[mask_L] - xL) / (lamb_L))
    Q[mask_C] = QL + N * (x[mask_C] - xL)
    Q[mask_R] = QR * np.exp(-(x[mask_R] - xR) / (lamb_R))
    
    if len(x) == 1:
        Q = Q[0]
    return Q


def hs_1d(x, N=None, kD=None,
          xL=None, xR=None, hL=None, hR=None, lamb_L=0., lamb_R=0.):
    """Return head in unconfined aquifer with leaky adjacent aquifers
    
    The aquifer has constant kD throughout. For xL < x< xR it is recharged by
    recharge N and for x < xL and x > xR the aquifer is leaky with the
    given values for lamb_L and lamb_R. Then both lamb_L and lamb_R are zero, 
    then the head will be fixed at hL and hR respectively at xL and xR.
    This head, therefore shows the head in a strip of land between xL and xR
    bounded on both sides to marshy or well-drained areas characterized by
    their characteristic length lambda, which can be different at the left and
    at the right.
    
    Parameters
    ----------
    x: np.ndarray or float
        coordinate where head is computed
    kD: float
        transmissivity
    xL, xR: floats
        Area boundary where elevation can be set.
    hL, hR: floats
        Left and right head, achieved when lambdas are zero.
    lamb_L, lamb_R: floats
        Characteristic length of left and right area.
    """
    L = xR - xL
    
    # Flow at left-hand area boundary (at x=xL)
    QL = Q_hs_1d(xL, N=N, kD=kD, xL=xL, xR=xR,
                 hL=hL, hR=hR, lamb_L=lamb_L, lamb_R=lamb_R)
    QR = QL + N * L
    
    print(f"QL = {QL:.4g}, QR = {QR:.4g}")
    
    # Head above hL and HR caused by QL and QR respectively
    dhR = +QR  * lamb_R / kD
    dhL = -QL  * lamb_L / kD
    
    x = np.atleast_1d(x).astype(float)
    h = np.zeros_like(x)
    
    # Deal with the three area separately
    mask_L = x < xL
    mask_R = x > xR
    mask_C = ~(mask_L | mask_R)
    
    # Center area
    h[mask_C] = (hL - QL * lamb_L / kD - QL / kD * (x[mask_C] - xL)
                 - N / (2 * kD) * (x[mask_C] - xL) ** 2)
    # Left area
    h[mask_L] = hL + dhL * np.exp(+(x[mask_L] - xL) / lamb_L)
    # Right area
    h[mask_R] = hR + dhR * np.exp(-(x[mask_R] - xR) / lamb_R)
    
    # If x is a float:
    if len(x) == 1:
        h=h[0]
    return h


# %% Analytical solution Bruggeman(1999) 370.01

def A(alpha, a, kD1, kD2, L1, L2):
    """Return function A(alpha) in Bruggeman 370_07.
    Parameters
    ----------
    alpha: integration parameter
    a: float a >= 0
        location of the well
    kD1, kD2 floats
        transmissivities for x < 0 and for x > 0
    L1, L2: floats
        L1 = np.sqrt(kD1 c1)
        L2 = np.sqrt(kD2 c2)
    """
    return np.exp(-a * np.sqrt(alpha ** 2 + 1 / L2 ** 2)) / (
        kD1 * np.sqrt(alpha ** 2 + 1 / L1 ** 2) + kD2 * np.sqrt(alpha ** 2 + 1 / L2 ** 2)
    )

def arg1(alpha, a, x, y, kD1, kD2, L1, L2):
    """Return integrand in formula for Phi1(x, y)"""
    return A(alpha, a, kD1, kD2, L1, L2) * np.exp(x * np.sqrt(alpha ** 2 + 1 / L1 ** 2)) * np.cos(y * alpha)

def arg2(alpha, a, x, y, kD1, kD2, L1, L2):
    """Return integrand for formula for Phi2(x, y)"""
    return A(alpha, a, kD1, kD2, L1, L2) * np.exp(-x * np.sqrt(alpha ** 2 + 1 / L2 ** 2)) * np.cos(y * alpha)

def brug370_01(X=None, Y=None, xw=None, Q=None, kD1=None, kD2=None, c1=None, c2=None):
    """Return drawdown problem 370_01 from Bruggeman(1999).
    
    The problem computes the steady drawdown in a well with extraction $Q$ at $xw=a, yw=0$
    in a leaky aquifer in which the properties $kD$ and $c$ jump at $x=0$ with
    $kD_1$ and $c_1$ for $x \le 04 and $kD_2$ and $c_2$ for $x \ge 0$.
    
    Parameters
    ----------
    X, Y: floats or arrays
        x, y coordinates
    xw: float
        Well x-coordinate (yw = 0)
    kD1, kD2, c1, c2: floats
        Aquifer transmissivities and aquitard resistances for $x \e 0$ and for $x \ge 0$ respectively.
    """
    # Convert scalars to arrays for vectorized computation
    X = np.atleast_1d(X).astype(float)
    if np.isscalar(Y):
        Y = Y * np.ones_like(X)
    assert np.all(X.shape == Y.shape), f"X.shape {X.shape} != Y.shape {Y.shape}"

    # If xw < 0, transform the problem and reuse function
    if xw < 0:
        X, Y, Phi = brug370_01(-X, Y, xw=-xw, Q=Q, kD1=kD2, kD2=kD1, c1=c2, c2=c1)
        return -X, Y, Phi

    # Compute characteristic length for $x<0$ and $x>0$
    L1, L2 = np.sqrt(kD1 * c1), np.sqrt(kD2 * c2)
    a = xw  # Well x-coordinate
    
    # Create output array for head Phi
    Phi = np.full_like(X, np.nan, dtype=np.float64)

    # Mask for points where x != a
    mask_x_a = X != a
    mask_x_neg = X < 0
    mask_x_pos = ~mask_x_neg
    
    # Evaluate `phi` for x < 0
    if np.any(mask_x_neg):
        valid = mask_x_a & mask_x_neg
        Phi[valid] = np.vectorize(
            lambda x, y: Q / np.pi * quad(arg1, 0, np.inf, args=(a, x, y, kD1, kD2, L1, L2))[0]
        )(X[valid], Y[valid])

    # Evaluate `phi` for x > 0    
    if np.any(mask_x_pos):
        valid = mask_x_a & mask_x_pos
        Phi[valid] = np.vectorize(
            lambda x, y: (
                Q / (2 * np.pi * kD2) * (K0(np.sqrt((x - a) ** 2 + y ** 2) / L2) -
                K0(np.sqrt((x + a) ** 2 + y ** 2) / L2))
                + Q / np.pi * quad(arg2, 0, np.inf, args=(a, x, y, kD1, kD2, L1, L2))[0]
            )
        )(X[valid], Y[valid])

    return X, Y, Phi


if __name__ == "__main__":
    
    # Example, Use of mirror wells, which immediately shows all 4 possibilities
    # The problem is an aquifer between xL and xR with as boundaries that the aquifer
    # at xL and or xR is fixed or closed.
    
    # Set aquifer properties
    kD, S, c= 600, 0.001, 200
    lambda_ = np.sqrt(kD * c)
    
    # Set system extension properties and coordinates
    xL, xR, xw, Nmirror= -200, 200, 100, 30
    Lclosed, Rclosed = False, False # right and or left size of strip closed?
    L = xR - xL
    n = 0
    x = np.linspace(xL - n * L, xR + n * L, 1201)
    
    # Set extraction
    Q = -1200.

    # Show result of well between two fixed-head boundaries (using mirror wells)
    md = Mirrors(xL, xR, xw=xw, N=Nmirror, Lclosed=Lclosed, Rclosed=Rclosed)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1 = md.show(ax=ax1, figsize=(8, 2), fcs=('yellow', 'orange'))

    s = np.zeros_like(x)
    
    # The drawdown of the well itself, which is at md.xw with sign md.sw
    r = np.sqrt((md.xw - x) ** 2)
    s = md.sw * Q / (2 * np.pi * kD) * K0( r / lambda_)
    
    # Add the drawdown due to mirror wells starting with the left one
    for xw, sgn in zip(md.xLD, md.sLD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)
        
    # Add the drawdown due to mirror well starting with the right one
    for xw, sgn in zip(md.xRD, md.sRD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)

    ax2.set_title(f"Verlaging door put met Q={Q} m3/d op xw={md.xw} m tussen randen met vaste $h$\n" +
    fr"op xL={xL} en xR={xR} m. kD={kD} m2/d, $\lambda$={lambda_:.0f} m, {Nmirror} maal gespiegeld.",
                 fontsize=10)
    ax2.set(xlabel="x [m]", ylabel='head [m]')
    ax2.plot(x, s, label=f"Q = {Q} m3/d")
    ax2.plot(xL, 0, 'ro', label='linker rand vast op 0.')
    ax2.plot(xR, 0, 'bo', label='rechter rand vast op 0.')
    ax2.grid()
    ax2.legend()
    # plt.show()
    
    # %% Example of using mirror ditches to simulate transient filling of a basin

    # Aquifer properties and coordinates
    kD, S = 600., 0.2
    xL, xR, AL, AR = -100., 100., 1.0, 1.0
    x = np.linspace(xL, xR, 201)
    
    b = (xR - xL) / 2 # Half width of the basin
    
    # Times for the simulation
    T50 = 0.28 * b ** 2 * S / kD # Halftime of the drainage of the basin
    ts = np.arange(7) * T50 # times to show
    ts[0] = 0.01 * T50 # First time should be > zero
    
    # Using xw=None in the call of Mirrors generates coords of mirror ditaches
    md = Mirrors(xL, xR, xw=None, N=30, Lclosed=False, Rclosed=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"Basin kD={kD} m2/d, S={S}, T50={T50:.4g} d")
    ax.set(xlabel="x [m]", ylabel="head [m]")
    
    for t in ts:
        s = np.zeros_like(x) # Initialize
        
         # Use xM to see what ditch coordinates are to the left
         # and what are to the right of the center of basin.
        xM = 0.5 * (xL + xR)
        
        # Ditches that started with the left one
        for xD, sgn in zip(md.xLD, md.sLD):            
            x_ = xD - x if xD >= xM else x - xD
            u = x_ * np.sqrt(S / (4 *  kD * t))
            s += sgn * AL * erfc(u)
            
        # Ditches that started with the right one
        for xD, sgn in zip(md.xRD, md.sRD):
            x_ = xD - x if xD >= xM else x - xD
            u = x_ * np.sqrt(S / (4 *  kD * t))
            s += sgn * AR * erfc(u)
            
        ax.plot(x, s, label=f"t = {t:.3f} = {t/T50:.4g} T50d")
    ax.grid()
    ax.legend(loc="lower right")
    # plt.show()


    # %% Testing the function "ground_surface"
    
    xL, xR = -2500., 2500.
    x = np.linspace(-4000., 4000., 801)
    
    # Use yM to set max elevation (approximately, adapt by trial and error)
    yM = 2.
    
    # Filter length in filtfilt. Adaept by trial and error to length of x
    Lfilt = 20
    
    # Set seed to an integer to get the same results each time.
    # Adept by trial and error.
    seed = 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f"Generated ground surface for xL={xL}, xR={xR}, yM={yM}, Lfilt={Lfilt}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Elevation [m]")
    
    h_grsurf = ground_surface(x, xL=xL, xR=xR, yM=yM, Lfilt=Lfilt, seed=1)
    
    ax.plot(x, h_grsurf, label=f"seed={seed}")
    ax.legend()
    ax.grid()
    # plt.show()


    # %% head in strip with adjacent leaky aquifers
    
    # Parameters for the cross section
    N, kD, xL, xR, hL, hR = 0.001, 600, -2500, 2500, 20, 22
    lamb_L, lamb_R = 100., 300.,

    kwargs = {'N': N, 'kD': kD, 'xL': xL, 'xR': xR, 'hL': hL, 'hR': hR}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Head in high area bounded by drained areas\n" +
                fr"kD={kwargs['kD']:.4g} m2/d, N={kwargs['N']:.4g} m/d, $\lambda_L$={lamb_L:.4g} m, $\lambda_R$={lamb_R:.4g} m")
    ax.set(xlabel="x [m]", ylabel="TAW [m]")
    ax.grid()

    ax.plot(x, ground_surface(x, xL=xL, xR=xR, yM=1, Lfilt=8) + hs_1d(x, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs), label='maaiveld')

    # Case with soft center-area boundaries
    ax.plot(x, hs_1d(x,  lamb_L=0.,     lamb_R=0.,     **kwargs), 
            label=f'h with hare area boundaries. N={kwargs['N']} m/d, kD={kwargs['kD']} m2/d, op xL={xL} m en xR={xR} m')

    # Case with hard center-area boundaries
    ax.plot(x, hs_1d(x,  lamb_L=lamb_L, lamb_R=lamb_R, **kwargs),
            label=fr'h with leaky adjacent areas. N={kwargs['N']} m/d, kD={kwargs['kD']} m2/d $\lambda_L$={lamb_L} m, $\lambda_R$={lamb_R} m')

    # Plot the boundary locations
    # Compute the head at the boundary locations
    hxL = hs_1d(xL, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)
    hxR = hs_1d(xR, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)
    
    ax.plot([xL, xR], [hxL, hxR], 'go',
            label=fr"Boundary be with leaky area with $\lambda_L$={lamb_L} en $\lambda_R$={lamb_R} m")
    
    # Annotate boundaries
    ax.annotate("Area boundary", xy=(xL, 20.), xytext=(xL, 24.5), ha='center',
                arrowprops={'arrowstyle': '-'})
    ax.annotate("Area boundary", xy=(xR, 20), xytext=(xR, 24.5), ha='center',
                arrowprops={'arrowstyle': '-'})

    # Mark the head at bouth boundaries with a red dot
    hxL = hs_1d(xL, lamb_L=0., lamb_R=0., **kwargs)
    hxR = hs_1d(xR, lamb_L=0., lamb_R=0., **kwargs)
    ax.plot([xL, xR], [hxL, hxR], 'ro', label=f"Hard boundary hL={hL}, hR={hR} m op  resp. x={xL} en x={xR}  m")


    # %% Bruggeman 370_01 example
    # Drawdown due to a well in a leaky aquifer were kD and c jump at x=0 
    
    # Aquifer parameters
    kD1, kD2, c1, c2 = 250., 1000., 250, 1000.
    L1, L2 = np.sqrt(kD1 * c1), np.sqrt(kD2 * c2)
    
    # Coordinates
    x = np.linspace(-2000, 2000, 401)
    y = 0

    # Well position
    xw = 200.

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.set_title("Bruggeman 370_01, verification, \n" +
                 fr"xw={xw:.4g} m, Qw={Q:.4g} m3/d\nkD1={kD1:.4g} m2/d, kD2={kD2:.4g} m2/d, c1={c1:.4g} d, c2={c2:.4g} d", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("phi [m]")
    ax.grid(True)
        

    # Show symmetrie for xw > 0 and xw < 0 by interchanging kD1 <-> kD2, c1 <-> c2 
    X, Y, Phi1 = brug370_01(x, y, xw=+xw, Q=Q, kD1=kD1, kD2=kD2, c1=c1, c2=c2)    
    X, Y, Phi2 = brug370_01(x, y, xw=-xw, Q=Q, kD1=kD2, kD2=kD1, c1=c2, c2=c1)

    # Compare Brug370_01 with De Glee leake aquifer solution
    # Using same kD2 and c2 should give De Glee
    X, Y, Phi3 = brug370_01(x, y, xw=xw, Q=Q, kD1=kD2, kD2=kD2, c1=c2, c2=c2)
    Phi4 = Q / (2 * np.pi * kD2) * K0(np.abs(x - xw)   / np.sqrt(kD2 * c2))

    # Using kD1 and c1 also just gives De Glee for the kD1 and c1
    X, Y, Phi5 = brug370_01(x, y, xw=xw, Q=Q, kD1=kD1, kD2=kD1, c1=c1, c2=c1)
    Phi6 = Q / (2 * np.pi * kD1) * K0(np.abs(x - xw)   / np.sqrt(kD1 * c1))

    # Show the combined results
    if X.ndim == 1:
        ax.plot(X, Phi1, label="Phi1")
        ax.plot(X, Phi2, label="Phi2")
        ax.plot(X, Phi3, label="Phi3, same properties (2) right")
        ax.plot(X, Phi4, '.', label=f"Phi4, K0, kD2={kD2}, c2={c2}")
        ax.plot(X, Phi6, label="Phi3, same properties (1) left")
        ax.plot(X, Phi6, '.', label=f"Phi4, K0, kD1={kD1}, c1={c1}")
    else: # If X and Y are 2D, then contour
        levels = 15
        CS = ax.contour(X, Y, Phi1, levels=levels)
        ax.clabel(CS, levels=CS.levels, fmt="{:.2f}")
        
    ax.legend(loc='best', fontsize='x-small')
    
    
    plt.show()


