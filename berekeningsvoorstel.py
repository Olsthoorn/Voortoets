# %% [markdown]
# # Voortoets, Module 3 - Inhoudelijke uitwerking van module 3 in de online toepassing van de voortoets: het bepalen van de reikwijdte van effecten voor de indirecte effectgroepen - Thema Grondwater
# 
# Auteurs:
# J. Bronders, J. Patyn, I.Van Keer, N. Desmet, J. Vos, W. Peelaerts, L. Decorte & A. Gobin
# 
# In opdracht van ANB, 2013

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from etc import newfig, attr
from scipy.signal import lfilter, filtfilt
from scipy.special import k0 as K0
from scipy.integrate import quad, quad_vec

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

# %% [markdown]
# De stroming Q [m2/d] is gelijk aan
# 
# $$Q	=-T\frac{dh}{dx}=N(x - x_L) + Q_L$$
# 
# Na integratie
# $$h(x)	= h_L + \frac{Q_{L}}{T} (x - x_L) -\frac{N}{2T}(x - x_L)^{2}$$
# 
# $$Q_{L}=\frac{T}{x_L - x_R} \left(h_R - h_L + \frac{N}{2 T}(x_R - x_L)^2\right)$$
# 
# $$Q_{L}=-\frac{T}{L} \left(h_R - h_L + \frac{N}{2 T}L^2\right)$$
# 
# 
# Met deze vergelijkingen kan de stijghoogte en de stroming worden berekend. Analytische uitwerking is niet nodig.

# %%
from matplotlib.path import Path
from matplotlib.patches import PathPatch

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

# Example, immediately shows all possibilities
kD, S, c= 600, 0.001, 200
lambda_ = np.sqrt(kD * c)

xL, xR, xw, N= -200, 200, 0, 30

_, ax = plt.subplots()
md = Mirror_ditches(xL, xR, xw=xw, N=N, Lclosed=False, Rclosed=False)
_ = md.show(ax=ax, figsize=(8, 2), fcs=('yellow', 'orange'))

Q = -1200.
n = 0
L = xR - xL
x = np.linspace(xL - n * L, xR + n * L, 1201)
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

# %%
from scipy.special import erfc

kD, S = 600., 0.2
xL, xR, AL, AR = -100., 100., 1.0, 1.0
b = (xR - xL) / 2
T50 = 0.28 * b ** 2 * S / kD

ts = np.arange(7) * T50
ts[0] = 0.01 * T50
x = np.linspace(xL, xR, 201)

md = Mirror_ditches(xL, xR, N=30, Lclosed=False, Rclosed=False)
_, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f"Basin kD={kD} m2/d, S={S}, T50={T50:.4g} d")
ax.set(xlabel="x [m]", ylabel="head [m]")
for t in ts:
    s = np.zeros_like(x)
    xM = 0.5 * (xL + xR)
    for xD, sgn in zip(md.xLD, md.sLD):
        x_ = xD - x if xD >= xM else x - xD
        u = x_ * np.sqrt(S / (4 *  kD * t))
        s += sgn * AL * erfc(u)
    for xD, sgn in zip(md.xRD, md.sRD):
        x_ = xD - x if xD >= xM else x - xD
        u = x_ * np.sqrt(S / (4 *  kD * t))
        s += sgn * AR * erfc(u)
    ax.plot(x, s, label=f"t = {t:.3f} = {t/T50:.4g} T50d")
ax.grid()
ax.legend(loc="lower right")
plt.show()

    

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt

def ground_surface(x, xL=None, xR=None, yM=1, Lfilt=20, seed=3):
    """Return a smooth ground surface elevation for visualization purposes.
    
    The elevation will be 0 at xL and xR and approach yM in the middle.

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

    # Apply smoothing and return elevation
    filtered = y * filtfilt(np.ones(Lfilt) / Lfilt, 1, z, method='gust')
    return filtered

# Test de functie
xL, xR, yM, Lfilt = -2500., 2500., 2., 20
x = np.linspace(-4000., 4000., 801)

plt.title(f"Generated ground surface for xL={xL}, xR={xR}, yM={yM}, Lfilt={Lfilt}")
plt.xlabel("x [m]")
plt.ylabel("Elevation [m]")
seed = 1
plt.plot(x, ground_surface(x, xL=xL, xR=xR, yM=yM, Lfilt=Lfilt, seed=1), label=f"seed={seed}")
plt.legend()
plt.grid()
plt.show()


# %%
N, kD = 0.001, 20 * 50
Q = -1200 # m3/d

xL, xR, hL, hR = -2500., 2500., 20., 22.

x = np.linspace(xL, xR, 5000)
y = np.zeros_like(x)

xM = 0.5 * (xL + xR)
xW, yW = xM - 1000., 0.

L = xR - xL
def h_1d(x, N=None, kD=None, xL=None, xR=None, hL=None, hR=None):
    """Return head for 1D flow between xL and xR for given hL and hR and N"""
    QL = -kD / (xR - xL) * (hR - hL + N / (2 * kD) * (xR - xL) ** 2)
    h = hL - QL / kD * (x - xL) - N / (2 * kD) * (x - xL) ** 2
    return h

h = h_1d(x, N=N, kD=kD, xL=xL, xR=xR, hL=hL, hR=hR)

_, ax = plt.subplots()
ax.set(title="Gebied tussen twee rechte randen met gefixeerde stijghoogte en onttrekking", xlabel="x [m]", ylabel="TAW [m]")
ax.grid(True)

ax.plot(x, h, label="zonder put, alleen N")
ax.plot(x[::10], ground_surface(x[::10], xL=xL, xR=xR, yM=1, Lfilt=8) + h[::10], label='maaiveld')

# Get the well positions
md = Mirror_ditches(xL=xL, xR=xR, xw=xW, N=30)
# md.show()

yld = yrd = 0.

# The real (actual well:
FQ = Q / (2 * np.pi * kD)
r = np.sqrt((x - md.xw) ** 2 + (y - yW) ** 2)

# We can also use leaky aquifer which ensures the drawdown approaches zero at infinity
lamb_ = 6000.
s = md.sw * FQ * K0(r / lamb_)

# Add the effect of all the mirror wells
for i, (xld, sld, xrd, sld) in enumerate(zip(md.xLD, md.sLD, md.xRD, md.sLD)):
    rRD = np.sqrt((x - xld) ** 2 + (y - yld) ** 2)
    rLD = np.sqrt((x - xrd) ** 2 + (y - yrd) ** 2)    
    s += +sld * FQ * K0(rRD / lamb_) # Don't need R here, because it cancels out
    s += +sld * FQ * K0(rLD / lamb_) # Don't need R here, because it cancels out
    
ax.plot(x, h + s, label=f"verlaagde grondwaterstand (K0, lambda = {lamb_:.0f})")
ax.annotate('Onttrekking', xy=(md.xw, 20), xytext=(md.xw, 24.5), ha='center', arrowprops={'arrowstyle': '<-', 'lw': 1.5} )
ax.legend(loc="lower right")

plt.gcf().savefig(os.path.join(dirs.images, "puntbron_hantush_strip_A.png"), transparent=True)

# %%
# Grondwaterstanden contourlijnen
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith('0'):
        s = f"{x:.0f}"
    return f"{s}"

x = np.linspace(-7000, 7000, 701)
y = np.linspace(-3000, 3000, 601)
X, Y = np.meshgrid(x, y)

kD, S, xL, xR, xw, yw = 600., 0.2, -2500, 2500., -1500., 0.

md = Mirror_ditches(xL=xL, xR=xR, xw=xw, N=100)

lamb = 6000. # Fixtieve zeer hoge spreidingslengte
r = np.sqrt((X - md.xw) ** 2 + (Y - yw) ** 2)

# Using K0 to make sure dd goes to zero for very large r
# With log, we don't get zero drawdown along both canals, only near y=0

s = md.sw * FQ * K0(r / lamb)
# Add the effect of all the mirror wells
for xl, sl, xr, sr in zip(md.xLD, md.sLD, md.xRD, md.sLD):
    yl, yr = 0., 0.   
    rRD = np.sqrt((X - xl) ** 2 + (Y - yl) ** 2)
    rLD = np.sqrt((X - xr) ** 2 + (Y - yr) ** 2)    
    s += sr * FQ * K0(rRD / lamb)
    s += sl * FQ * K0(rLD / lamb)
    
# Set levels to contour
levels = hL + np.arange(0, 6., 0.2)

ax = newfig("Contourlijnen van de grwst. verlaging", "x [m]", "y [m]")
ax.plot(X[Y[:, 0]==0][0], s[Y[:,0]==0][0]) # Well

ax.annotate('Left ditch', [xL, -1], xytext=[xL, 1],
            ha='center', arrowprops={'arrowstyle': '-', 'lw': 0.75, 'color': 'black'})
ax.annotate('Right ditch', [xR, -1], xytext=[xR, 1],
            ha='center', arrowprops={'arrowstyle': '-', 'lw': 0.75, 'color': 'black'})
ax.annotate('Extraction', xy=(md.xw, -1.5), xytext=(md.xw, 1.5),
            ha='center', arrowprops={'arrowstyle': '<-', 'lw': 3, 'color': 'red'})
ax.annotate('Mirror well L1', xy=(md.xLD[0], 1.0), xytext=(md.xLD[0], 1.5),
            ha='center', arrowprops={'arrowstyle': '->', 'lw': 3, 'color': 'green'})
ax.annotate('Mirror well R1', xy=(md.xRD[0], 1.0), xytext=(md.xRD[0], 1.5),
            ha='center', arrowprops={'arrowstyle': '->', 'lw': 3, 'color': 'green'})

ax.set_ylim(-2, 2) # Ensure arrows are completely visible.
plt.show()

ax = newfig("Contourlijnen van de grwst. verlaging", "x [m]", "y [m]")
ax.plot(md.xw, yw, 'ro') # Well
ax.vlines([xL, xR], y[0], y[-1], color='b', lw=1) # The canals

ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])

ax.set_aspect(1.0)

levels = np.arange(-3, 3.2, 0.05)

CS = ax.contour(X, Y, s, levels=levels)
ax.clabel(CS, CS.levels, fmt=fmt, fontsize=10)

ax.plot(md.xw, yw, 'ro') # Well
ax.vlines([xL, xR], y[0], y[-1], color='b', lw=1) # The canals

plt.gcf().savefig(os.path.join(dirs.images, "puntbron_dupuit_strip_C.png"), transparent=True)
plt.show()


# %% [markdown]
# # Bruggeman, een gebied gaat over in een ander gebied.
# 
# Oplossing 370.01 van Bruggmman (1999) beschrijft de verlaging door een put in een semi-gespannen pakket waarvan de eigenschappen kD en c verspringen op de lijn x=0. Deze oplossing biedt de mogelijk om de invloed te berekenen van een put nabij de grens van een hoog droog naar een lager nat gebied. De oplossing is nuttig omdat in veel situaties de grens tussen hoge niet gedraineerde en lage goed gedrainneerde gebieden niet scherp zal zijn begrensd door een diep kanaal of een diepe sloot danwel een rivier of een beek. In dat geval werkt de verlaging die buiten een laag (meestal meer natuurlijk)gebied wordt veroorzaakt door tot in het lagere gebied. Men kan al naar gelang de omstandigheden het enige gebied een zeer hoge weerstand geven, zodat het verschil tussen een watertafel aquifer in het enige gebied of een semi-gespannen aquifer verwaarloosbaar is, terwijl aan de andere zijde van de grens wel sprake is van een duideijke lek.
# 
# Voor $x<0$ geldt $k_2D_1$ en $c_1$ zonder onttrekking, terwijl voor $x>0$ $k_2D_2$ en $c_2$ gelden met onttrekking $Q$ op $x=a$. Op $x=0$ geldt dat de stijghoogte voor $x<0$, $\phi_1$ gelijk is aan die voor $x>0$, $\phi_2$ en dat het specifiek debiet in de $x$-richting  $q_x$ daar continue is.
# 
# De stijghoogte voor $x \le 0$, $\phi_1(x, y)$, is
# 
# $$
# \phi_1(x,y) = \frac Q \pi \intop_0^\infty A(\alpha) 
# \exp
# \left(
#     x \sqrt{
#         \alpha^2 + \frac{1}{\lambda_1^2}
#         }
#   \right)
#  \cos(y \alpha) d\alpha
# $$
# 
# De stijghoogte voor $x \ge 0$, $\phi_2(x, y)$ is
# $$
# \phi_2(x, y) = \frac Q {2 \pi k_2D_2} \left[
#     K_0 \left\{\frac 1 {\lambda_2} \sqrt{(x - a)^2 + y^2} \right\} +
#     K_0 \left\{\frac 1 {\lambda_2} \sqrt{(x + a)^2 + y^2} \right\}
#      \right] +
# $$
# $$
# +\frac Q \pi \intop_0^\infty A(\alpha) \exp \left( -x \sqrt{\alpha^2 + \frac 1 {\lambda_2^2}} \right)
# $$
# met
# $$
# A(\alpha) = \frac{\exp \left(-a \sqrt{\alpha^2 + \frac 1 {\lambda_2^2}}\right)}
# {k_1D_1 \sqrt{\alpha^2 + \frac 1 {\lambda_1^2}} + k_2D_2 \sqrt{\alpha^2 + \frac 1 {\lambda_2^2}}}
# $$
# 
# De integralen zijn probleemloos numeriek te integreren, waarmee de functie gemakkelijk te berekenen is. Deze berekening wordt hieronder geïmplementeerd. Waarbij ook de situatie wordt meegenomen waarin de put links van $x = 0$ staat, m.a.w. wanneer $a <0$. Dit omkeren wordt geïmplenteerd door de coordinaten om te draaien.

# %%
import numpy as np
from scipy.integrate import quad    # numerieke integratie
from scipy.special import k0 as K0  # Modified Bessel function of the second kind

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
    
    Parameters
    ----------
    X, Y: floats or arrays
        x, y coordinates
    xw: float
        Well x-coordinate (y = 0)
    kD1, kD2, c1, c2: floats
        Aquifer transmissivities and aquitard resistances for x < 0 and for x > 0
    """
    # Convert scalars to arrays for vectorized computation
    X = np.atleast_1d(X)
    if np.isscalar(Y):
        Y = Y * np.ones_like(X)
    assert np.all(X.shape == Y.shape), f"X.shape {X.shape} != Y.shape {Y.shape}"

    # If xw < 0, transform the problem and reuse function
    if xw < 0:
        X, Y, Phi = brug370_01(-X, Y, xw=-xw, Q=Q, kD1=kD2, kD2=kD1, c1=c2, c2=c1)
        return -X, Y, Phi

    # Compute length scales
    L1, L2 = np.sqrt(kD1 * c1), np.sqrt(kD2 * c2)
    a = xw  # Well x-coordinate
    
    # Create output array for Phi
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

# Test examples
kD1, kD2, c1, c2 = 250., 1000., 250, 1000.
L1, L2 = np.sqrt(kD1 * c1), np.sqrt(kD2 * c2)
a = 200.

try:
    ax = newfig(f"Bruggeman 370_01, verification, xw={xw:.4g} m, Qw={Q:.4g} m3/d\nkD1={kD1:.4g} m2/d, kD2={kD2:.4g} m2/d, c1={c1:.4g} d, c2={c2:.4g} d",
            "x", "phi")
except Exception as e:
    print(f"Error: {e}")
    breakpoint()
    
x = np.linspace(-2000, 2000, 401)
y = 0
xw = 200.

# symmetrie for xw > 0 and xw < 0
X, Y, Phi1 = brug370_01(x, y, xw=xw, Q=Q, kD1=kD1, kD2=kD2, c1=c1, c2=c2)
try :
    X, Y, Phi2 = brug370_01(x, y, xw=-xw, Q=Q, kD1=kD2, kD2=kD1, c1=c2, c2=c1)
except Exception as e:
    print(f"Error {e}")
    breakpoint()

# Using same kD and c gives just De Glee
X, Y, Phi3 = brug370_01(x, y, xw=xw, Q=Q, kD1=kD2, kD2=kD2, c1=c2, c2=c2)

# Compare with De Glee
Phi4 = Q / (2 * np.pi * kD2) * K0(np.abs(x - xw)   / np.sqrt(kD2 * c2))

# Using same kD and c gives just De Glee
X, Y, Phi5 = brug370_01(x, y, xw=xw, Q=Q, kD1=kD1, kD2=kD1, c1=c1, c2=c1)
Phi6 = Q / (2 * np.pi * kD1) * K0(np.abs(x - xw)   / np.sqrt(kD1 * c1))

if X.ndim == 1:
    ax.plot(X, Phi1, label="Phi1")
    ax.plot(X, Phi2, label="Phi2")
    ax.plot(X, Phi3, label="Phi3, same properties (2) right")
    ax.plot(X, Phi4, '.', label=f"Phi4, K0, kD2={kD2}, c2={c2}")
    ax.plot(X, Phi6, label="Phi3, same properties (1) left")
    ax.plot(X, Phi6, '.', label=f"Phi4, K0, kD1={kD1}, c1={c1}")
else:
    CS = ax.contour(X, Y, Phi1, levels=levels)
    ax.clabel(CS, levels=CS.levels, fmt="{:.2f}")
    
ax.legend()


# %% [markdown]
# # Example Bruggeman 370_01
# 
# Beschouw een hoger gebied dat grenst aan een lager gebied waarin de afvoer wordt gekarakteriseerd door zijn drainageweerstaand.
# 
# 
# $$ h = h_L - \frac N {2 kD} (x - x_L)^2 - \frac{Q_L}{kD}(x - x_L) $$
# 
# Op overgang naar het gebied met drainageweerstand rechts van $x_R$ moet $Q_R$ continu zijn
# $$ Q_L + N L = h_R \frac kD \lambda$$
# Zodat
# $$h(x_R) = h_R + (Q_L + N L) \frac \lambda kD$$
# 
# Ingevuld levert dit $Q_L$ op
# 
# $$Q_L = \frac{ -kD \left( h_R - h_L + \frac{N}{2 kD} (x_R - x_L)^2 +\frac{N}{kD} \lambda (x_R-x_L) \right)}
# {\lambda + x_R - x_L}$$
# 
# $$Q_L = \frac{ -kD \left( h_R - h_L + \frac{N}{2 kD}L^2 +\frac{N}{kD} \lambda L \right)}
# {\lambda + L}$$
# 
# $$Q_L = \frac{ -\frac{kD}{L} \left( h_R - h_L + \frac{N}{2 kD}L^2 +\frac{N}{kD} \lambda L \right)}
# {1 + \frac{\lambda}{L}}$$
# 
# Hieronder volgt een voorbeeld. Het is een doorsende tussen $x_L$ en $x_R$ waarbij $h_L$ en $h_R$ gegeven zijn met uniforme neerslag dat op $x_L% en $x_R$ aansluit op een gebied met een drainageweerstand.

# %%
N, kD = 0.001, 20 * 50.
Q = -3600 # m3/d

xL, xR, hL, hR = -2500., 2500., 20., 22.
lamb_L, lamb_R = 100., 300
x = np.linspace(xL -1000., xR + 1000., 701)
y = np.zeros_like(x)

xM = 0.5 * (xL + xR)
xW, yW = xM - 1000., 0.

X = np.arange(xL, xR + 1001, 10)
h = np.zeros_like(X)

def Q_hs_1d(x, N=None, kD=None, xL=None, xR=None, hL=None, hR=None, lamb_L=0., lamb_R=0., eps=1e-6):
    L = xR - xL
    QL = -kD / (L + lamb_L + lamb_R)  * (hR - hL + N / (2 * kD) * L ** 2 + N / kD * L * lamb_R)
    
    x = np.atleast_1d(x)
    Q = np.zeros_like(x)
    Q[x < xR] = QL + N * (x[x < xR] - xL)
    Q[x >= xR] = (QL + N *(xR - xL)) * np.exp(-(x[x >=xR] - xR) / (lamb + eps))
    if len(Q) == 1:
        Q = Q[0]
    return Q

def hs_1d(x, N=None, kD=None, xL=None, xR=None, hL=None, hR=None, lamb_L=0., lamb_R=0., eps=1e-6):
    L = xR - xL
    QL = Q_hs_1d(xL, N=N, kD=kD, xL=xL, xR=xR, hL=hL, hR=hR, lamb_L=lamb_L, lamb_R=lamb_R)
    QR = QL + N * L
    dhR = +QR  * lamb_R / kD
    dhL = -QL  * lamb_L / kD
    x = np.atleast_1d(x)
    h = np.zeros_like(x)
    mask_L = x <= xL
    mask_R = x >= xR
    mask_C = ~(mask_L | mask_R)
    h[mask_C] = hL - QL * lamb_L / kD - QL / kD * (x[mask_C] - xL)  - N / (2 * kD) * (x[mask_C] - xL) ** 2
    h[mask_L] = hL + dhL * np.exp(+(x[mask_L] - xL) / (lamb_L + eps))
    h[mask_R] = hR + dhR * np.exp(-(x[mask_R] - xR) / (lamb_R + eps))
    if len(x) == 1:
        h=h[0]
    return h

kwargs = {'N': N, 'kD': kD, 'xL': xL, 'xR': xR, 'hL': hL, 'hR': hR}

_, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Stijghoogte in hoog gebied grenzend aan laag met drainage\n" +
            fr"kD={kwargs['kD']:.4g} m2/d, N={kwargs['N']:.4g} m/d, $\lambda_L$={lamb_L:.4g} m, $\lambda_R$={lamb_R:.4g} m")
ax.set(xlabel="x [m]", ylabel="TAW [m]")
ax.grid()

ax.plot(x, ground_surface(x, xL=xL, xR=xR, yM=1, Lfilt=8) + hs_1d(x, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs), label='maaiveld')

ax.plot(x, hs_1d(x,  lamb_L=lamb_L, lamb_R=lamb_R, **kwargs),
        label=fr'h met zachte gebiedsrand. N={kwargs['N']} m/d, kD={kwargs['kD']} m2/d $\lambda_L$={lamb_L} m, $\lambda_R$={lamb_R} m')
ax.plot(x, hs_1d(x,  lamb_L=0.,     lamb_R=0.,     **kwargs), 
        label=f'h met harde gebiedsrand. N={kwargs['N']} m/d, kD={kwargs['kD']} m2/d, op xL={xL} m en xR={xR} m')

# Plot the boundary locations
hxL = hs_1d(xL, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)
hxR = hs_1d(xR, lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)
ax.plot([xL, xR], [hxL, hxR], 'go',
        label=fr"overgang hoog gebied naar laag kwelgebied met $\lambda_L$={lamb_L} en $\lambda_R$={lamb_R} m")
ax.annotate("gebiedsrand", xy=(xL, 20.), xytext=(xL, 24.5), ha='center',
            arrowprops={'arrowstyle': '-'})
ax.annotate("gebiedsrand", xy=(xR, 20), xytext=(xR, 24.5), ha='center',
            arrowprops={'arrowstyle': '-'})

# plot for both lamb_L = 0. and lamb_R = 0.
hxL = hs_1d(xL, lamb_L=0., lamb_R=0., **kwargs)
hxR = hs_1d(xR, lamb_L=0., lamb_R=0., **kwargs)
ax.plot([xL, xR], [hxL, hxR], 'ro', label=f"harde grens hL={hL}, hR={hR} m op  resp x={xL} en x={xR}  m")

lambda_ = 3000
s = brug370_01(x - xR, y, xw=xw - xR, Q=Q,
               kD1=kD, kD2=kD, c1=lambda_ ** 2 / kD, c2=lamb_R ** 2 / kD)
ax.plot(x, s[-1] + hs_1d(x,  lamb_L=lamb_L, lamb_R=lamb_R, **kwargs), '-',
        label=f'h + put acc. Bruggeman, Q={Q} m3/d, ' +
        fr"kD={kD} m2/d, $\lambda_L$={lamb_L}, $\lambda_R$={lamb_R}")

# Check with drawdown according to De Glee
h = hs_1d(x,  lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)

r = np.sqrt((x - xw) ** 2)
ax.plot(x, h + Q / (2 * np.pi * kD) * K0(r / lambda_), 'black', lw=1,
        label=fr"De Glee, Q={Q} m3/d, kD={kD} m2/d, $\lambda$={lambda_}")

ax.legend(loc="lower center", fontsize='x-small')

# %% Checking Bruggeman with 1D
# Check with drawdown according to De Glee

Q = -3600.
xw = 2000.
x = np.linspace(xL -1000., xR + 1000., 6001)
lamb_L, lamb_R = 3000, 300
kwargs.update(hL=0., hR=0.)

_, ax = plt.subplots()
ax.grid()
ax.set_title("Checking Brug370_01\n" + fr"Q={Q} m3/d kD={kD} m2/d, $\lambda_L$={lamb_L} , $\lambda_R$={lamb_R} ")
ax.set_ylabel("head [m]")

# Drawdown acc. to De Glee
r = np.sqrt((x - xw) ** 2)
ax.plot(x, Q / (2 * np.pi * kD) * K0(r / lambda_), 'black', lw=1,
        label=fr"De Glee Q={Q} m3/d, kD={kD} m2/d, $\lambda$={lamb_L} m")

# Drawdown acc. to Bruggeman solution 370_01
s = brug370_01(x - xR, y, xw=xw - xR, Q=Q, kD1=kD, kD2=kD,
               c1=lamb_L ** 2 / kD, c2=lamb_R ** 2 / kD)
ax.plot(x, s[-1], '-', color='red', label=fr"Brug, kD={kD} m2/d $\lambda_L$={lamb_L} m, $\lambda_R$={lamb_R} m")

# Head change due to recharge
h = hs_1d(x,  lamb_L=lamb_L, lamb_R=lamb_R, **kwargs)
ax.plot(x, h, label=fr"h, N={N} m/d, $\lambda_L$={lamb_L} m, $\lambda_R$={lamb_R} m")
ax.vlines([xL, xR], -2, 6, color='black')

ax.legend(loc="best", fontsize='xx-small')

plt.show()


# %% [markdown]
# ## Berekening wijziging grondwaterstand per type bron (primair effect)
# 
# ### Puntbronnen
# 
# Bepaling van de grondwaterstandsverandering ten gevolge van puntbronnen (pompputten).
# 
# De formule van Duipuit geldt voor stationaire verlaging in een pakket met horizontale bodem, uniforme doorlatendheid en een rand op afstand $R$ waarop of waarbuiten geen verlaging meer optreedt. Alleen wanneer in het beschouwde gebied een neerslagoverschot $N$ wordt via drainage of een dicht stelsel van waterlopen wordt afgevoerd en deze afvoer door de onttrekking wegvalt, kan men stellen dat deze door de verlaging weggevallen drainage ten goede komt aan het grondwater. In dat geval is de formule van Verruijt toepasbaar, althans tot hoogstens een afstand waarop het ingevangen neerslagoverschot gelijk is aan de onttrekking. De afstand tot waar de formule van Verruijt geldt is zelfs kleiner, en reikt namelijk niet verder dan de afstand waarop de afvoer door de drainage juist gelijk is aan nul. Daarbuiten voert de drainage of het oppervlaktestelsel nog steeds een deel van het neerslagoverschot af en geldt feitelijk de formule van De Glee, waarin het watervoerend pakket wordt gevoedt in evenredigheid met de verlaging, waarbij de drainageweerstand de evenredigheidsconstante vormt. De formule van Verruijt en die van De Glee worden op deze grens aan elkaar geknoopt. De combinatie heet de Formule van Blom, wat geen formule is maar twee formules die met elkaar zijn verbonden op de afstand waarop de drainage juist gelijk aan nul is. Dit is waar de verlaging gelijk is aan $N c$ met $N$ het neerslagoverschot eventueel vermeerderd met kwel en $c$ de drainageweerstand.
# 
# De formule van Dupuit geldt wanneer er geen terugkoppeling is tussen verlaging en voeding van het watervoerend pakket. Dit is het geval in een gebied met weinig of geen oppervlaktewater of drainages. De formule van Dupuit kan natuurlijk op een bepaalde afstand overgaan in de formule van Verruijt en nog weer verder van de put in die van De Glee. Dit is afhankelijk van de grondwaterafvoer naar het oppervlak in het beschouwde gebied.
# 
# Het is intussen best ingewikkeld om 3 analytische formules aan elkaar te knopen, zodat uit praktische overwegingen ofwel Dupuit en De Glee danwel Verruijt en De Glee aan elkaar kunnen worden geknoopt afhankleijk van het gebied waarin men verkeert.
# 
# Het is intussen zeer de vraag in hoeverre een stationaire, jaargemiddelde situatie en verlaging relevant is voor de te beschermen SBZ. Immers de situatie in de winter met neerslag zonder vardamping en de zomer met een neerslagtekort zijn zeer verschillend. In de zomersituatie met neerslagtekort zullen drainages en waterlopen geheel droog kunnen raken. Onder zulke omstandigheden beweegt het grondwateroppervlak geheel vrij van het drainage- en het slotenstelsel. De verlaging zal zich onder zulke omstandigheden dan ook voortdurend uitbreiden overeenkomstig de formule van Theis. Dit gaat door totdat als gevolg van het neerslagoverschot in het najaar de grondwaterstanden weer zover zijn gestegen dat drainages en of sloten weer grondwater, in feite neerslagoverschot plus mogelijk kwel, gaan afvoeren. Men zou daarom moeten stellen dat voor de berekening van de verlaging in de zomer periode in het algemeen uitgegaan moet worden van de formule van Theis. Deze bepaalt de maximale uitbreiding aan het eind van de zomer. Deze invloedsstraal krimpt vervolgens in de periode met neerslagoverschot, waarvoor een stationaire berekening veel meer voor de hand ligt. Het hierbij te gebruiken neerslagoverschot kan 1 tot 2 mm/dag zijn.
# 
# De berekening van de verlaging met de stationaire formules zijn derhalve eerder geschikt voor het berekenen van de verlaging in de winter en de formule van Theis voor niet-staionaire verlaging voor de berekening van de verlaging aan het eind van de zomer. Hierbij moet een keuze worden gemaakt voor de duur van de zomer. Hiervoor kan een prakische grens van 100 dagen worden gekozen.
# 
# De in het rapport gebruikte formules van Dupuit en Verruijt houden beide rekening met de afname van de natte dikte van het watervoerende pakket als gevolg van de verlaging. Deze formules zijn daarom niet-lineair. Echter, de extra verlaging als gevolg van de afname van de effectieve dikte van het watervoerende pakket is alleen relevant in de nabijheid van de pompput. Voor het berekenen van de invloed op een SBZ op enige afstand is het effect niet van belang, waardoor de formules van Dupuit en Verruijt met vaste pakketdikte eveneens toepasbaar zijn. De formules van De Glee en van Theis hebben zelf geen variant met variabele laagdikte.
# 
# De drainageweerstand $c$ [d] is een belangrijke factor bij de bepaling van de invloedsstraal in de stationaire situatie. Deze is gelijk aan $c = \overline{h} / N$ waarbij $\overline{h}$ de gemiddelde stijghoogte is boven het drainageniveau danwel het peil in het stelsel van waterlopen. De verlaging waarbij de formule van Verruijt overgaat in die van De Glee is $\overline{h} = N c$.
# 
# Voor de niet-stationaire situatie in de zomer, zodat de formule van Theis geldt, is de invloedsstraal
# $$ R = \sqrt{\frac{2.25 \, kD\,t}{S} }$$
# 
# De genoemde analytische formules zijn allemaal axiaal-symmetrisch. Dit impliceert dat zij alleen van toepassing zijn wanneer de omstandigheden in alle richtingen hetzelfde zijn waardoor de verlagingslijnen cirkels vormen. Dit is in de werkelijkheid lang niet overal het geval. Bijvoorbeeld wanneer zich aan een of meer zijden belangrijke waterlopen bevinden of wanneer de ingreep plaats vindt in een overgangsgebied, bij bijvoorbeeld een droog zandgebied aan een zijde dat aan de andere zijde geleidelijk overgaat in een natter gebied met een steeds dichter stelsel van waterlopen en of drainages. De situatie wordt dan al snel te complex voor een Voortoets die terecht gebruiksvriendelijkheid, robuustheid en daarmee eenvoud op de voorgrond stelt. Op zulke situaties zijn axiaal-summetrische formules alleen in uitzonderingen toepasbaar door randvoorwaarden te simuleren middels superpositie met spiegelputten. In de meest gevallen zullen zij feitelijk niet (goed) toepasbaar zijn. De formueles voor stationaie verlaging geven wanneer toegepast onder de geschetste omstandigheden in overgangsgebieden bovendien een onderschatting van de werkelijke verlaging. Voor de niet-stationaire verlaging volgens Theis is dit niet het geval. Het is kortom van belang te onderkennen wanneer de axiaal-symmetrische analytische formules wel of juist niet toepasbaar zijn.

# %% [markdown]
# # Voorbeeld van de toepassing van formules voor stationaire verlaging en vergelijking met de niet-stationaire verlaging volgens Theis

# %% [markdown]
# ## Oppervlaktebron bouwput
# 
# Bemaling voor een bouwput heeft tot doel een bepaalde verlaging van de grondwaterstand binnend de bouwput te realiseren. Dus niet het onttrokken grondwater maar de te realiseren verlaging binnen de bouwput is hier het doel. De initiatiefnemer wordt nu gevaagd het onttrekkingsdebiet op te geven. Hij kan dit alleen in redelijkheid doen, wanneer hij de bouwput al met grondwaterformules of een grondwatermodel heeft doorgerekend. Het is bovendien de vraag op het opgegeven debiet juist is. In plaats van de verantwoordlijkheid voor het opgeven van het debiet althans voor de Voortoets bij de initiatiefnemer te leggen, kan dit ook binnen de Voortoets worden berekend op basis van de verlaging die de initiatiefnemer wenst te bereiken. Alleen in dat geval zal het berekende effect op de grondwaterstand consistent zijn met de laageigenschappen die in de Voortoets voor de berekening worden gebruikt. Eventueel kun de berekening worden gebruikt om het opgegeven debiet te controleren. Maar zelf uitrekenen heeft mogelijk de voorkeur in verband met de vereiste consistentie en de aanvraag en de beoordeling daarvan onafhankelijk maken van wat de initiatiefnemer opgeeft.
# 
# Wanneer de bouwput opgevat kan worden als een cirkelvorige verlaging, iets dat op enige afstand van de bouwput praktisch altijd het geval zal zijn, kan de situatie buiten de bouwput met een enkele put in het centrum van de bouwput worden berekend. Voor de verlaging buiten de bouwput maakt het niet uit of de bemaling wordt berekend met een onttrekkringsscherm langs de bouwput of met een enkele put in de bouwput. Dit is exact het geval voor de formules voor stationaire verlaging en na enige tijd ook voor de formules voor tijdsafhankelijke verlaging.
# 
# Het staat de intiatiefnemer natuurlijk vrij om de bouwput als een enkele cirkelvormige verlaging te beschouwen of als een combinatie van meerdere cikelvormige verlagingen.
# 
# Voor een bouwput geldt net als voor een puntonttrekking dat de effecten afhankelijk zijn van het jaargetijde. Om deze reden zou voor een bemaling in de zomer dan ook het beste niet-stationair het effect op de grondwaterstand kunnen worden berekend met de formule van Theis, terwijl voor de wintersituatie eerder de formules van Dupuit, Verruijt of Blom toepasbaar zullen zijn. Opnieuw is de vraag wat de onttrekkingsduur moet zijn om de zomersituatie te berekenen. Van mei t/m september, de periode met verdampingsoverschot is 150 dagen. In de huidige Voortoets wordt hier een periode van 100 dagen (ruim 3 maanden) voor aangehouden.

# %% [markdown]
# # Lijn- en sleufonttrekkingen en andere lijnbronnen
# 
# Lijn of sleufonttrekkingen associeren met met bemalingen voor bijvoorbeeld aanleg van leidingen waarvoor ontgraven moet worden tot beneden de watertafel. Deze   lijnvormige verlagingen hebben een aanzienlijke lengte en verplaatsesn in de tijd naarmate de werkzaamheden vorderen. Kenmerkend is hun lengte en de onttrekking met vele kleine bemalingsbronnen op enkele tot hoogstens enkele tientallen meters onderlinge afstand, die zijn aansloten op een vacuümpomp.
# 
# Andere lijnbronnen zijn sloten waarvan het peil wordt veranderd. Zulke peilveranderingen zijn een strikt andere situatie, maar de formules voor het berekenen van de grondwaterstanden zijn dezelfde.
# 
# Een eenvoudige formule van de verlaging is een variant van Dupuit, Verruijt of Blom voor de stationaire situatie. Deze kunnen worden toegepast, maar vergen een keuze ten aanzien van de randvoorwaarde, de afstand waarop de verlaging nul blijft, of waarop de verlaging het omringende oppervlaktewater droog trekt. Deze keuzen zijn lastig te maken, waarom dan ook meestal in de praktijk en in de huidige versie van de Voortoets de tijdsafhankelijke verlaging door de sleufonttrekking wordt berekend, temeer daar zulke voor werkzaamheden uitgevoerde bemalingen sowieso in de tijd beperkt zijn. Dit is uiteraard anders bij permanente peilveranderingen.
# De niet-stationaire verlagign door onttrekking langs een oneindig lange lijn is
# $$h(x) = h_0 \, \mathtt{erfc}\left(x \sqrt{\frac{S_y}{4 kD t}}{}\right)$$
# 
# $$q(x t) = h_0 \, kD\, \frac{2}{\pi}\,\exp\left(-\frac{x^2 S_y}{4 kD t}\right)\sqrt{\frac{S}{4 kD t}}=h_0 \, \sqrt{\frac{kD S}{\pi t}} \exp\left(\frac{x^2 S}{4 kD t}\right)$$
# $$q(0, t) = \sqrt{\frac{kD S}{\pi t}}$$
# 
# Deze formule houdt geen rekening met de afname van de dikte van het watervoerend pakket als gevolg van de verlaging. Dit hoeft niet erg te zijn, omdat de formule een bovengrens oplevert voor het onttrokken debiet en dus voor de verlaging op afstand van de sleuf. Ook houdt de fromule geen rekening met de lengte van de sleuf, waardoor de uitstraling van de verlaging naar beide einden van de sleuf toe minder wordt. Hier geen rekening mee houden leidt dus eveneens tot een bovengrens van de verlaging. In de praktijk kan de verlaging die met deze formule wordt berekend aan het einden van de sleuf worden gevormd door dezelfde verlaging aan beide zijden van de sleuf met cirkels met elkaar te verbinden.
# 
# Voor de duur van de ingreep geldt de echte tijdsduur tot een maximum van 365 dagen, wat als conservatief wordt beschouwd.
# 
# Evenals voor de pompputten en bouwputten zou hier voor de winterperiode een stationaire verlaging kunnen worden berekend met de eendimensionale versie van Verruijt en De Glee vanaf het punt waar de verlaging gelijk is aan $N c$, de afstand waarop de afvoer door de sloten juist nul is, terwijl voor de zomer de niet-stationaire situatie aangehouden zou kunnen worden met een duur gelijk aan de lengte van de zomer waarin gemiddelde een verdampingsoverschot aanwezig is. Voor het berekenen van een stationaire wintersituatie is een neerslagoverschot en een drainageweerstand noodzakelijk.
# 
# ## Vlakdekkende drainage (drainage met parallelle drains)
# 
# Het gaat erom de invloed van de drainage buiten het te draineren terrein. De verlaging binnen het terrein wordt niet beschouwd. Onder deze uitgangspunten kunnen dezelfde lijn- of sleufformules worden toegepast die hiervoor al zijn besproken, waarbij als locatie de randen van het te draineren terrein worden aangeduid.
# 
# De terrein of gebiedsdrainage zal mogelijk alleen in het natte halfjaar werken, wanneer de grondwaterstanden normaliter hoog oplopen. In dat geval is alleen een stationare berekening voor de wintersituatie nodig. De invloed op de zomer bestaat dan uit een grondwaterstand die aan het begin van de zomer is verlaagd door de drainage, waardoor het grondwaterverloop gedurende de zomer verlaagd blijft ten opzichte van de oorspronkelijke situatie. In een kwelgebied kan een drainage ook gedurende de zomer grondwater afvoeren, waardoor het grondwater voortdurend is verlaagd tot het drainageniveau. De afvoer van de drainage ademt in dat geval mee met het verloop van het neerslagoverschot.
# 
# Voro de stroming vanaf de waterscheiding, 4x=0$, dus waar het verhang gelijk is aan nul geldt
# $$ q -kh\frac{dh}{dx}= N x$$
# 
# Na integratie
# $$ h^2 = -\frac{N}{k} x^2 + C$$
# De integratie constante volgt uit de eis dat voor $x=B$ de waterdiepte in de aquifer gelijk is aan $H$. Na invulling levert dit
# $$h^2 - H^2 = \frac N k (B^2 - x^2)$$
# 
# De maximale waterdiepte in de aquifer voor $x=0$, met $h_0^2 - H^2 = (h_0 - H)(h_0 + H) = \Delta h (\Delta h + 2 H)$ volgt
# 
# $$B^2 = \frac k N \Delta h (\Delta h + 2 H)$$
# 
# of
# 
# $$ B = \sqrt{\frac k N (2 H \, \Delta h + \Delta h^2)} $$
# 
# De formule in Bonders et al. (2013) geeft zonder afleiding is
# 
# $$ B = \sqrt{L^2 + \frac k N {2 H \, \Delta h + \Delta h ^2}} - L$$
# 
# en is delzelfde behalve de variabele $L$, de breedte van het te draineren gebied. Zowel wanneer $L$ in deze formule naar nul of naar oneindig gaat, gaat de formule over in de hierboven afgeleide versie zonder $L$. De variabele $L$ aangeduide als de breedte van het gebied waar geen verlaging (door de drainage?) optreedt heeft geen basis in de fysica van de grondwaterstroming. In werkelijkheid rijkt de verlaging tot $x=B4 indien de verlaging het neerslagoverschot naar de verlaagde sloot afleidt of tot een randvoorwaarde verder naar rechts, waar de grondwatestand wordt gefixeerd. In dat laatste geval, ook omdat de maximale verlaging bij de sloot $\Delta h << H$, kan de verlaging eenvoudig lineair worden genomen tussen die bij de sloot en nul bij de lijn waar de grondwaterstand is gefixeerd
# 
# $$ h - H = \Delta h \, \frac{x-L}{L} $$
# 
# 
# 
# 
# De formule voor de afstand $B$ waarover de watertafel is verlaagd wordt door Bonders et al. opgegeven als
# 
# $$ (B + L)^2 = \frac{2 kD \Delta h + k \Delta h^2}{N} + L^2$$
# 
# Ik heb nog even geen idee waar deze formule precies vandaan komt.
# 
# 
# 
# 
# 

# %% [markdown]
# 


