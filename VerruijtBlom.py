# %% [markdown]
# # Analytische oefeningen t.b.v de Voortoets
# 
# @Theo Olsthoorn (12-03-2024 .. 30-11-2024)
# 
# # Intro, verantwoording
# 
# Na de vorige vergadering van 12 maart 2024 heb ik een aantal aspecten geanalyseerd die van belang zouden kunnen zijn voor een relatief eenvoudige, analytische analyse van de impact van een ingreep op een grondwatersysteem. De analyse heb ik uitgevoerd en gedocumenteerd in voorliggend Jupyter (Python) notebook, dat uitleg en code bevat om het uitgelegde te kwantificeren en te laten zien.
# 
# Voor een aantal complexere concepten zoals vertraagde nalevering, debietverloop van een bemaling met constante verlaging, hoe lang het duurt voor de verlaging stationair wordt e.d. zijn vereenvoudigingen voorgesteld die in de praktijk goed zullen werken.
# 
# Voor een zinvolle analyse van een fysische ingreep in het grondwatersysteem, zoals een nieuwe puntonttrekking, bemaling of verandering van de loop of het niveau van oppervlaktewater, is een beeld nodig van de opbouw van de ondergrond en van de drainage in het gebied (de randvoorwaarden). Bovendien is men geïnteresseerd in de overlap van de impact met bijzondere beschermingsgebieden. Deze drie vormen van informatie zijn gebonden aan beschikbare kaarten zoals die van de habitatgebieden, het oppervlaktewater, en bodemlagen. De laatste twee zijn aanwezig in de kaarten waarop het grondwatermodel van Vlaanderen is gebaseerd. Deze kaarten bevatten ook de benodigde informatie over de waarden van de bodemconstanten, zoals doorlaatvermogen, weerstand tussen lagen en bergingscoëfficiënten. Mogelijk kunnen op basis van de beschikbare laagverbreidingskaarten ook complexere situaties worden gesignaleerd, zoals breuken en andere scherpere overgangen tussen gebieden,  die niet met eenvoudige formules kunnen worden geanalyseerd of waarvoor een aangepaste berekening kan worden voorgesteld of voorgeschreven. Het uit kaarten halen van de randvoorwaarden voor een berekening is vermoedelijk het meest complex of valt in de praktijk het meest op af te dingen. Uiteraard kunnen randvoorwaarden worden voorgeschreven zoals een invloedsradius of duur van de onttrekking waarmee moet worden gerekend, zoals dat nu reeds in de Voortoets het geval is.
# 
# Voor de benodigde onderliggende gegevens moet er i de Voortoets toegreep zijn tot het Vlaamse grondwatermodel, of althans de kaarten waar dit op gebaseerd is, zodat deze kaarten als een ruimtelijke database kunnen worden beschouwd en bevrraagd op bodemconstanten die voor een gegeven locatie moeten worden gebruikt.
# 
# Door op een dergelijke manier de voor elke vraag benodigde informatie op te vragen, kan de onderliggende machinerie van de voortoets steeds gemakkelijk worden aangepast en verbeterd naar nieuwe inzichten.
# 
# Het resultaat van de Voortoets zol op deze wijze ook steeds in lijn zijn met de eventueel naderhand uit te voeren bredere analyse, waar dan zonodig een ruimtelijk grondwatermodel aan te pas komt, dat immers op dezelfde gegevens is gebaseerd. Een kernpunt zou dus moeten zijn dat de gegevens die gebruikt worden in de voortoets dezelfde zijn als die in het ruimtelijke model van Vlaanderen, waarbij de voortoetsberekeningen zich zullen baseren op de ondergrond gegevens ter plaatse van de ingreep en het ruimtelijk model met de ruimtelijke variatie rekening houdt. 
# 
# De info-vraag van de Voortoets zal altijd zeer beperkt zijn, zodat de ICT-belasting navenant laag blijft en de informatie dus real-time over het internet (via een URL) moet kunnen worden opgevraagd en worden opgezocht op de ruimtelijke kaarten.
# 

# %% [markdown]
# # Imports voor de benodigde functionaliteit

# %%
import os
import sys
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy.special import exp1 as Wt, exp1
from scipy.special import k0 as K0, k1 as K1
from scipy.integrate import quad
import pandas as pd
from itertools import cycle
from importlib import reload

# %% [markdown]
# # Basisfuncties die verderop worden gebruikt:

# %% Theis and Hantush functions

def newfig(title, xlabel, ylabel, xlim=None, ylim=None, xscale=None, yscale=None, figsize=None):
    """Set up a new figure with a single axes and return the axes."""
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True)
    return ax


def Wh(u, rho=0):
    """Return Hantush's well function values.
    
    Parameters
    ----------
    u = r^s / (4 kD t)
    rho = r / lambda,    lambda = sqrt(kD c)
    
    >>>Wh(0.004, 0.03)
    4.894104204671381    
    """
    def integrant(y, rho):
        """Return the function to be integrated."""
        return np.exp(-y - (rho / 2) ** 2 / y) / y
    
    def w(u, rho): # Integrate the argument
        return quad(integrant, u, np.inf, args=(rho,))[0]
    
    wh = np.vectorize(w) # Vectorize function w(u, rho) so we can use arrays as input.
    
    return np.asarray(wh(u, rho), dtype=float)

Wh(0.004, 0.03)


def Wb(tau, rho=0):
    """Return Hantush well function values using the Bruggeman (1999) form of Hantush's Well function.
    
    Bruggeman (1999) prefers uses a somewhat different form of Hantush's Well function, one in which
    time and distance to the well are truly separated. However, this loses a bit the connection with
    Theis, which we have always been used to. Bruggeman uses W(tau=t / (cS), rho=r / lambda)
    instead of W(u=r^2 S / (4 kD t), rho=r / lambda)
    
    Parameters
    ----------
    tau = t/ (cS)
    rho = r / lambda, lambda= sqrt(kD c)
    
    >>>Wb(0.05625, 0.03)
    4.894104204671381    
    """
    def integrant(x, rho):
        """Return the function to be integrated."""
        return np.exp(-x - (rho / 2) ** 2 / x) / x
    
    def w(tau, rho):
        """Integrate the argument."""
        return quad(integrant, 0, tau, args=(rho,))[0]
    
    # Vectorize function w(u, rho) so we can use arrays as input.
    wh = np.vectorize(w)
    return np.asarray(wh(tau, rho), dtype=float)


def Wb1(tau, rho=0):
    """Return Hantush well function values using the Bruggeman (1999) form of Hantush's Well function.
    
    This function just converts the input paramters of Bruggean into
    those of the regular Hantush well function. It's the simplest conversion,
    no extra functionality is needed this way.
    
    Parameters
    ----------
    tau = t/ (cS)
    rho = r / lambda, lambda= sqrt(kD c)
    
    >>>Wb(0.05625, 0.03)
    4.894104204671381    
    """
    u = rho ** 2  / (4 * tau)
    return Wh(u, rho)

def SRtheis(T=None, S=None, r=None, t=None):
    """Return Step Responss for the Theis well function."""
    dt = np.diff(t)
    assert(np.all(np.isclose(dt[0], dt))), "all dt must be the same."
    u = r ** 2 * S / (4  * T * t[1:])
    return np.hstack((0, 1 / (4 * np.pi * T) * exp1(u)))

def BRtheis(T=None, S=None, r=None, t=None):
    """Return Block Response for the Theis well function"""
    SR = SRtheis(T, S, r, t)
    return np.hstack((0, SR[1:] - SR[:-1]))

def IRtheis(T=None, S=None, r=None, t=None):
    dt = np.diff(t)
    assert np.all(np.isclose(dt[0], dt))
    u = np.hstack((np.nan, r ** 2 * S / (4 * T * t[1:])))
    return np.hstack((0, 1 / (4 * np.pi * T) *  np.exp(-u[1:]) / t[1:])) * dt[0]

#u = np.logspace(-4, 1, 51)
u, rho = 0.004, 0.03
print("Wh(u={:.4g}, rho={:.4g}) = {}".format(u, rho, Wh(u, rho)))
tau = rho ** 2 / (4 * u)
print("Wb(tau={:.4g}, rho={:.4g}) = {}".format(tau, rho, Wb(tau, rho)))


# %% Wells
class WellBase(ABC):
    """Base class for several well functions"""
        
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        """Initialize a well.
        
        Parameters
        ----------
        xw, yw: float
            Well location coordinates.
        rw: float, Optional
            well radius
        z1, z2: float, optional
            Depth boundaries of the well.
        aqprops: dict, optional
            Aquifer propertiers (see below).
            
        Returns
        -------
        a member of the Well class
        """
        if not (np.isscalar(xw) and np.isscalar(yw)):
            raise ValueError("xw and yw must both be scalars.")
        self.xw = xw
        self.yw = yw
        self.rw = rw
        self.z1 = z1
        self.z2 = z2
        self.aq = aqprops
        return
    
    @abstractmethod
    def dd(self, *args, **kwargs):
        """Return drawdown for well."""
        pass
    
    @abstractmethod
    def h(self, *args, **kwargs):
        """Return head in water table aquifer."""
        pass
    
    @abstractmethod
    def Qr(self, *args, **kwargs):
        """Return the discharge across a circle with radius r."""
        pass
    
    @abstractmethod
    def qxqy(self, *args, **kwargs):
        """Return specific discharge vectors qx and qy."""
        pass

    
    @staticmethod    
    def itimize(a):
        """Return a as a scalar if a is an array and len(a) == 1"""
        if isinstance(a, np.ndarray) and len(a) == 1:
            a = a.item()
        return a

    @staticmethod
    def check_xyt(x, y, t):
        """Return checke x, y and t.
        
        x and y must both be floats of t is an array and vice versa.
        If x and y are arrays they must have the same shape.
        y may be None and in which case it is ignored,
        """
        if isinstance(t, np.ndarray):
            if not np.isscalar(x):
                raise ValueError("x must be scalar if t is an array!")
            if y is not None:
                if not np.isscalar(y):
                    raise ValueError("y must be a scalar if x is a scalar!")
        else:
            x, y = WellBase.check_xy(x, y)
        return x, y, t
    
    @staticmethod
    def check_xy(x, y):
        """Return checked coordinates x and y
        
        x and y are turned into arrays.
        The must have the same shape.
        y will be ignored if None
        """
        x = np.atleast_1d(x).astype(float)
        if y is not None:
            y = np.atleast_1d(y).astype(float)
            if not np.all(x.shape == y.shape):
                raise ValueError("x.shape must equal y.shape!")
        return x, y
    
    @staticmethod
    def check_keys(subset, aqprops):
        """Check and update the aquifer properties according to subset.
        
        Subset is the set of required keys. It is checked if these are
        in the dict aqprops. If not it is tried to update aqprops. If
        that is not possible a ValueError is raised.
        """
        if {'k', 'D'}.issubset(aqprops):
            aqprops['kD'] = aqprops['k'] * aqprops['D']
        if 'kD' not in aqprops.keys():
            raise ValueError("Neither kD, k nor D in aqprops!")

        if 'S' in subset and 'S' not in aqprops.keys():
            raise ValueError("S not in aqprops!")
        
        # kD is guaranteed so make c and lambda consistent if either c or lambda in aqprops
        if 'c' in aqprops.keys():
            aqprops['lambda'] = np.sqrt(aqprops['kD'] * aqprops['c'])
        if 'lambda' in aqprops.keys():
            aqprops['c'] = aqprops['kD'] ** 2  / aqprops['lambda']
                              
        # Then if c or lambda is required by subset but still not in aqprops --> ValueError  
        if 'c' in subset:
            if 'c' not in aqprops.keys():
                    raise ValueError("Neither c nor lambda in aqprops")
            
        if 'lambda' in subset:
            if 'lambda' not in aqprops.keys():
                    raise ValueError("Neither c nor lambda in aqprops!")
                
    
    def radius(self, x=None, y=None):
        """Return the distance between well and x, and y.
        
        Parameters
        ----------
            x : np.ndarray
                x-coordinates
            y: np.ndarray like x or None
                y-coordinates or None, when only x is used
        """
        x = np.atleast_1d(x).astype(float)
        if y is None:
            r = np.sqrt((self.xw - x) ** 2)
        else:
            y = np.atleast_1d(y).astype(float)
            if not np.all(x.shape == y.shape):
                raise ValueError("x and y must have same shape")
            try:                
                r = np.sqrt((self.xw - x) ** 2 + (self.yw - y) ** 2)
            except Exception as e:
                print(e)
                raise
        
        r = WellBase.itimize(r)
        return r
    
    
class wTheis(WellBase):
    """Class for handling drawdown and other calcuations according to Theis"""
    
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD, S'}, self.aq)
        return
    
    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for Theis well type.")


    def dd(self, Q=None, x=None, y=None, t=None):
        """Return well's drawdown according to Theis.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.
        
        Parameters
        ----------
        Q: float
            constant extraction by the well
        x: float or np.ndarray of floats
            x coordinate[s] of drawdown point.
        y: None or float or np.ndarray
            y coordinate[s] or None
        t: float or np.ndarray
            times to compute dd (all dd[t <= 0] =0)        
        """
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        
        FQ = Q / (4 * np.pi * self.aq['kD'])
        
        if isinstance(t, np.ndarray):
            u = np.zeros_like(t)
            s = np.zeros_like(u)
            u = r ** 2 * self.aqprops['S']  / 4 * self.aq['kD'] * t[t > 0]
            s[t > 0] = FQ * Wt(u)            
        else:
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)
            s = FQ * Wt(u)
        
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None, y=None, t=None):
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        if isinstance(t, np.ndarray):
            Qr_ = np.zeros_like(t) + Q
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t[t > 0])
            Qr_[t > 0] = Q * np.exp(-u)
        else:
            Qr_ = np.zeros_like(r) + Q
            u = r[r > 0] ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)
            Qr_[r > 0] = Q * np.exp(-u)
            
        Qr_ = WellBase.itimize(Qr_)        
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)


class wHantush(WellBase):
    """Clas for computing drawdown and other values according to Hantush."""
    
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD', 'c'}, self.aq)

    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for Hantush well type.")

    
    def dd(self, Q=None, x=None, y=None, t=None):
        """Return well's drawdown according to Theis.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.
        
        We use Bruggemans function Wb(tau, rho) because this separates
        time and disrance with u and rho not.
        
        Parameters
        ----------
        Q: float
            constant extraction by the well
        x: float or np.ndarray of floats
            x coordinate[s] of drawdown point.
        y: None or float or np.ndarray
            y coordinate[s] or None
        t: float or np.ndarray
            times to compute dd (all dd[t <= 0] =0)        
        """
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
                            
        FQ = Q / (4 * np.pi * self.aq['kD'])
        
        if isinstance(t, np.ndarray):
            s = np.zeros_like(t)
            tau = t[t > 0] / (self.aq['c'] * self.aq['S'])
            rho = r  / self.aq['lambda']
            s[t > 0] = FQ * Wb(tau, rho)            
        else:
            tau = t / (self.aq['c'] * self.aq['S'])
            rho = r  / self.aq['lambda']
            s = FQ * Wb(tau, rho)
        
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None, y=None, t=None, dr=1e-3):
        """Return the Qr for the Hantush well.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.

        We do this numerically because there is no simple analytical solution for it.        
        """        
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        
        tau = t / (self.aq['c'] * self.aq['S'])                                                
        if isinstance(t, np.ndarray):
            Qr_ = np.zeros_like(t)
            Qr_[tau > 0] = Q * (Wb(tau[t > 0], (r + dr) / self.aq['lambda']) -
                       Wb(tau[t > 0], (r - dr) / self.aq['lambda'])) * r  / dr
        else:
            Qr_ = np.zeros_like(r)            
            Qr_[r > 0] = Q * (Wb(tau, (r[r > 0] + dr) / self.aq['lambda'] -
                        Wb(tau, (r[r > 0] - dr) / self.aq['lambda']))) * r[r > 0]  / dr
        
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
    

class wDupuit(WellBase):
    
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD'}, self.aq)
        return

    def h(self, Q=None, x=None, y=None, R=None):
        """Return wetted aquifer thickness according to Dupuit.
        
        This is the same as Verruijt for N=0.        
        """
        WellBase.check_keys({'k', 'D'}, self.aq)
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)

        H = self.aq['D']
        
        FQ = Q / (np.pi * self.aq['k'])
        
        h2 = H ** 2 * np.ones_like(r)        
        h2[r >= R] = H ** 2  - FQ * np.log(R / r[r >= R])

        h = np.sqrt(h2)

        if len(h) == 1:        
            h = h.ravel()[0]
        return h

    def dd(self, Q=None, x=None, y=None, R=None, N=None):
        """Return drawdown according to Dupuit.
        
        This is the same as Verrijt for N=0.        
        """
        x, y = WellBase.check_xyt(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s = np.zeros_like(r)
        s = FQ * np.log(R / r[r <= R])

        if len(s) == 1:
            s = s.ravel()[0]        
        return s
    
    def Qr(self, Q=None, N=None, x=None, y=None):
        """Return the flow at a distance given by x and y.
        
        This is the same as Verrijt for N=0 --> just Q.
        """
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        
        Qr_ = np.zeros_like(r)
        Qr_[r <= R] = Q
        
        Qr_ = WellBase.itemize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        qx, qy = WellBase.qxqy(Q, self.xw, self.yw, x, y)
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def divide(self, Q=None, N=None):
        """Return radius of water divide."""
        return np.NaN     

class wDeGlee(WellBase):
    """Axial symmetric flow to well in semi-confiende aqufier according to De Glee.
    
    This the same as Blom for r > [dd == Nc]
    
    """
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD', 'c'}, self.aq)
        return
    
    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for De Glee well type.")


    def dd(self, Q=None, x=None, y=None):
        """Return drawdown using uniform kD."""
        WellBase.check_keys({'kD', 'lambda'}, self.aq)
        x, y = WellBase.check_xyt(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s =   FQ * K0(r[r >= R] / self.aq['lambda'])
        s = WellBase.itimize(s)
        return s
        
    def Qr(self, Q=None, x=None,  y=None):
        """Return Q(r)"""
        x, y = WellBase.check_xyt(x, y)
        
        r = self.radius(x, y)
        Qr_ = Q / (2 * np.pi * self.aq['lambda']) * K1(r / self.aq['lambda'])
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
        

class Brug370_1(WellBase):

    def __init__(self, xw=None, yw=None,  rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        if not {'kD1', 'kD2' 'c1', 'c2'}.issubset(self.aq.keys()):
            raise ValueError("Missing one or more of kD1, kD2, c1, k2")

    @classmethod
    def A(cls, alpha, a, kD1, kD2, L1, L2):
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

    @classmethod
    def arg1(cls, alpha, a, x, y, kD1, kD2, L1, L2):
        """Return integrand in formula for Phi1(x, y)"""
        return cls.A(alpha, a, kD1, kD2, L1, L2) * np.exp(x * np.sqrt(alpha ** 2 + 1 / L1 ** 2)) * np.cos(y * alpha)

    @classmethod
    def arg2(cls, alpha, a, x, y, kD1, kD2, L1, L2):
        """Return integrand for formula for Phi2(x, y)"""
        return cls.A(alpha, a, kD1, kD2, L1, L2) * np.exp(-x * np.sqrt(alpha ** 2 + 1 / L2 ** 2)) * np.cos(y * alpha)

    def dd(self, Q=None, x=None, y=None, xw=None):
        """Return drawdown problem 370_01 from Bruggeman(1999).
        
        The problem computes the steady drawdown in a well with extraction $Q$ at $xw=a, yw=0$
        in a leaky aquifer in which the properties $kD$ and $c$ jump at $x=0$ with
        $kD_1$ and $c_1$ for $x <= 0$ and $kD_2$ and $c_2$ for $x <= 0$.
        
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
        
        X, Y = WellBase.check_xy(x, y)
        if Y is None:
            Y = np.zeros_like(X)
        
        # If xw < 0, transform the problem and reuse function
        if xw < 0:
            aqprops = {}
            for p1, p2 in zip(['kD1', 'kD2', 'c', 'c2'], ['kD2', 'kD1', 'c2', 'c1']):
                aqprops[p1] = self.aq[p2]
            w = Brug370_1(xw=self.xw, yw=self.yw,  rw=self.rw,
                          z1=self.z1, z2=self.z2, aqprops=aqprops)
                
            X, Y, Phi = w.dd(Q=Q, x=-X, y=Y, aqprops=aqprops)
            return -X, Y, Phi

        # Compute characteristic length for $x<0$ and $x>0$
        L1 = np.sqrt(self.aq['kD1'] * self.aq['c1'])
        L2 = np.sqrt(self.aq['kD2'] * self.aq['c2'])
        
        # Create output array for head Phi
        Phi = np.full_like(X, np.nan, dtype=np.float64)

        # Mask for points where x != a
        mask_x_a = X != self.xw
        mask_x_neg = X < 0
        mask_x_pos = ~mask_x_neg
        
        # Evaluate `phi` for x < 0
        if np.any(mask_x_neg):
            valid = mask_x_a & mask_x_neg
            Phi[valid] = np.vectorize(
                lambda x, y: Q / np.pi * quad(self.__class_.arg1, 0, np.inf,
                    args=(self.xw, x, y, self.aq['kD1'], self.aq['kD2'], L1, L2))[0]
            )(X[valid], Y[valid])

        # Evaluate `phi` for x > 0    
        if np.any(mask_x_pos):
            valid = mask_x_a & mask_x_pos
            Phi[valid] = np.vectorize(
                lambda x, y: (
                    Q / (2 * np.pi * self.aq['kD2']) * (K0(np.sqrt((x - self.xw) ** 2 + y ** 2) / L2) -
                    K0(np.sqrt((x + self.xw) ** 2 + y ** 2) / L2))
                    + Q / np.pi * quad(self.__class__.arg2, 0, np.inf,
                            args=(self.xw, x, y, self.aq['kD1'], self.aq['kD2'], L1, L2))[0]
                )
            )(X[valid], Y[valid])

        return X, Y, Phi
    
    def h(self, Q=None, X=None, Y=None, xw=None):
        raise NotImplementedError("h not implemented for Brug370_1")
    
    def Qr(self, Q=None, X=None, Y=None, xw=None):
        raise NotImplementedError("Qr not implemented for Brug370_1")

    def qxqy(self, Q=None, X=None, Y=None, xw=None):
        raise NotImplementedError("qxqy is not implemented for Brug370_1")


class wVerruijt(WellBase):
    """Axial symmetric flow according to Verruijt and Blom.
    
    Verruijt is Dupuit + Recharge N for r <= R
        
    """
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD'}, self.aq)
        return

    def h(self, Q=None, x=None, y=None, R=None, N=None):
        """Return wetted aquifer thickness according to Verruijt."""
        WellBase.check_keys({'k', 'D'}, self.aq)
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)

        H = self.aq['D']
        
        FQ = Q / (np.pi * self.aq['k'])
        
        h2 = H ** 2 * np.ones_like(r)        
        h2[r >= R] = H ** 2  - FQ * np.log(R / r[r >= R]) +\
            N / (2 * self.aq['k']) * (R ** 2 - r[r >= R] ** 2)

        h = np.sqrt(h2)
        h = WellBase.itimize(h)
        return h

    def dd(self, Q=None, x=None, y=None, R=None, N=None):
        """Return drawdown according to Verrijt."""
        x, y = WellBase.check_xyt(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s = np.zeros_like(r)
        s = FQ * np.log(R / r[r <= R]) - N / (4 * self.aq['kD']) * (R **2 - r[r <= R] ** 2)
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, N=None, x=None, y=None):
        """Return the flow at a distance given by x and y."""
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        
        Qr_ = np.zeros_like(r)
        Qr_[r <= R] = Q - np.pi * N * r[r <= R] ** 2
        
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def divide(self, Q=None, N=None):
        """Return radius of water divide."""        
        r = np.sqrt(Q / (np.pi * N))
        return r if r < self.R else np.nan
             
   
class wBlom(WellBase):
    """Axial symmetric flow according to Verruijt and Blom.
    
    Blom equals Verruijt for R < R[dd == Nc] and DeGlee for R >= R[dd == Nc]
    
    """
    def __init__(self, xw=0., yw=0., rw=None, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD', 'c'}, self.aq)
        return

    def h(self, Q=None, x=None, y=None, N=None, r=None):
        """Returng h using 'k' and 'D' has Hinf."""
        WellBase.check_keys({'k', 'D'}, self.aq)
        x, y = WellBase.check_xyt(x, y)
        r = self.radius(x, y)
        R = self.getR(Q=Q, N=N, R=R)
        RL = R  / self.aq['lambda']
        QR = Q - np.pi * R **2 * N     
        Hinf = self.D
        HR = Hinf - N * self.aq['c']
        
        h = np.zeros_like(r) + HR
        h[h < RL] = np.sqrt(HR ** 2 + N / (2 * self.k) * (R ** 2 - r[r < RL] ** 2)
                            - Q / (np.pi * self.k) * np.log(R / r[r < RL]))

        h[r >= R] = Hinf - QR / (2 * np.pi * self.kD) *\
            K0(r[r >= R] / self.aq['lambda']) / (RL * K1(RL))
            
        h = WellBase.itimize(h)
        return h


    def dd(self, Q=None, x=None, y=None, N=None):
        """Return drawdown using uniform kD."""
        x, y = WellBase.check_xyt(x, y)
        
        r = self.radius(x, y)
        
        R = self.getR(Q=Q, N=N, R=self.R)
        RL = R / self.aq['lambda']
        QR = Q - np.pi * R ** 2 * N
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s = np.zeros_like(r)
        s[r < RL] =  (FQ * np.log(R / r[r < RL])
                - N / (4 * self.aq['kD']) * (R ** 2 - r[r < RL] ** 2) + N * self.aq['c'])
        s[r >= R] = (QR / (2 * np.pi * self.aq['kD']) * 
                     K0(r[r >= R] / self.aq['lambda']) / (RL * K1(RL)))
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None,  y=None, N=None):
        """Return Q(r)"""
        x, y = WellBase.check_xyt(x, y)
        
        r = self.radius(x, y)

        R = self.getR(Q=Q, N=N, r=x.mean()) # use x.mean() as start for Newton iterations.
        QR = Q - np.pi * R ** 2 * N        
        R_lam = R / self.aq['lambda']
        r_lam = r / self.aq['lambda']
        Qr_ = np.zeros_like(r)
        Qr_[r < R]  = Q - np.pi * r ** 2 * N
        Qr_[r >= R] = QR * r_lam[r >= R] * K1(r_lam[r >= R]) / K1(R_lam)
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def rdivh(self, Q=None, N=None):
        """Return location of water divide (freatic)."""
        R = self.getR(Q=Q, N=N, R=1.0)
        rD = np.sqrt(Q / (2 * np.pi * N))
        return rD if rD <= R else np.nan
    
    def rdivD(self, Q=None, N=None):
        """Return location of water divide (fixed D)."""
        R = self.getR(Q=Q, N=N, R=1.0)        
        rD = np.sqrt(Q / (np.pi * N))
        return rD if rD <= R else np.nan
    
    def y(self, R=1.0, Q=None, N=None):
        """Return y for Newton Raphson detetermination of R."""
        QR = Q - np.pi * R ** 2 * N
        Rl = R  / self.aq['lambda']
        return -N * self.aq['c'] + QR / (2 * np.pi * self.aq['kD']) * K0(Rl) / (Rl * K1(Rl))
    
    def y1(self, R=None, Q=None, N=None):
        """Return dy(R)/dR for Newton Raphson determination of R."""        
        RL = R / self.aq['lambda']
        k0k1 = K0(RL) / K1(RL)        
        return -N * self.aq['lambda'] / self.aq['kD'] * k0k1 -\
            (Q / R / (2 * np.pi * self.aq['kD'])  - N * R / (2 * self.aq['kD'])) *\
                (1 - k0k1  ** 2)

    # Newton Raphson functions
    def dydR(self, R=None, Q=None, N=None, dr=0.1):
        """Return numerical derivative of y in Newton iterations."""
        return (self.y(R=R + 0.5 * dr, Q=Q, N=N) -
                self.y(R=R - 0.5 * dr, Q=Q, N=N)) / dr
                
    def getR(self, Q=None, N=None, R=1.0, tolR=0.1, n=50,  verbose=False):
        """Return R by Newton's method."""
        if verbose:
            print(f"R initial={R:.3g} m")
        for i in range(n):
            dR = -self.y(R=R, Q=Q, N=N) / self.y1(R=R, Q=Q, N=N)
            R += dR
            if verbose:
                print(f"iteration {i}, R={R:.3g} m, dR = {dR:.3g} m")
            if abs(dR) < tolR:
                if verbose:
                    print(f"R final ={R:.3g} m")
                self.R = R
                return R
        R = np.nan
        self.R = R        
        print(f"R final ={R:.3g} m")
        return R
    
    def plot_newton_progress(self, R=None, Q=None, N=None, figsize=(10, 6)):
        """Plot progress of the Newton method to find R at which the drawdown is Nc.
        
        Parameters
        ----------
        R: float
            initial R to start iteration.
        Q: float
            well extraction
        N: float
            recharge
        """
        r = np.linspace(0, 300, 101)[1:]
        title = f"y(R) and progress of Newton iterations for Q={Q:.4g} , N={N:.4g}, c={self.aq['c']:.4g}"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=title, xlabel="r [m]", ylabel="y(r) [m]")
        
        ax.plot(r, self.y(R=r, Q=Q, N=N), label='y(r)')
        
        for _ in range(20):
            #y, y1 = self.y(R=R, Q=Q, N=N), self.dydR(R=R, Q=Q, N=N)     
            y, y1 = self.y(R=R, Q=Q, N=N), self.y1(R=R, Q=Q, N=N)
            ax.plot([R, R], [0, y], 'k')     
            ax.plot(R, y, 'bo', mfc='none')
            dR = - y / y1
            ax.plot([R, R + dR], [y, 0], 'k')            
            R += dR
            if abs(dR) < 0.1:
                break            
        ax.legend()
        return ax
        
    def plot_derivative_of_y(self, R=None, Q=None, N=None, figsize=(10, 6)):
        """Plot the derivative of y, analytical and numerically calculated."""
        r = np.linspace(0, 300, 301)[1:]        
        title = f"Derivatives of y for Q={Q:.4g} , N={N:.4g}, c={self.aq['c']:.4g}"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=title, xlabel="R [m]", ylabel="y(r) [m]")
        ax.plot(r, self.y1(  R=r, Q=Q, N=N), '-', label='y1 (analytic)')        
        ax.plot(r, self.dydR(R=r, Q=Q, N=N), '.', label='dydR (numerical)')
        ax.legend()
        return ax


class StripBase(ABC):
    """Base class for several 1D groundwater flow functions."""
        
    def __init__(self, aqprops={}):
        """Initialyze a well.
        
        Parameters
        ----------
        aqprops: dict
            Aquifer propertiers (see below)
            
        """
        self.aq = aqprops

    @abstractmethod
    def h(self, *args, **kwargs):
        """Return the drawdown."""
        pass

    @abstractmethod
    def dd(self, *args, **kwargs):
        """Return the drawdown."""
        pass

    @abstractmethod
    def Qx(self, *args, **kwargs):
        """Return the dicharge at x values."""
        pass
    
    @abstractmethod
    def h(self, *args, **kwargs):
        """Return head in water table aquifer."""
        pass
    
    @staticmethod
    def check_x(x=None):
        """Return x as float array for values >= 0"""       
        x = np.atleast_1d(x).astype(float)
        return x[x >= 0]
        
    @staticmethod
    def check_xt(x=None, t=None):
        """Return x, t making sure x is sclar if t is array and vice versa."""
        if isinstance(t, np.ndarray):
            t = t[t >= 0]
            if not np.isscalar(x):
                raise ValueError("x must be scalar if t is an array!")
        else:
            x = StripBase.check_x(x)
        return x, t
    

class Verruijt1D(StripBase):
    """Verruijt 1D solution"""
    
    def __init__(self, aqprops={}):
        super().__init__(aqprops=aqprops)
        
        WellBase.check_keys({'kD'}, self.aq)
        return


    def dd(self, Q=None, x=None, L=None, N=None):
        """Return drawdown (using confined kD)."""
        x = super().check_x(x)
        s = np.zeros_like(x)
        s[x < L] = (Q / self.aq['kD'] * (L - x[x < L]) -
                    N / (2 * self.aq['kD']) * (L **2 - x[x < L] ** 2))
        if len(s) == 1:
            s = s.ravel()[0]
        return s
    
    def Qx(self, Q=None, x=None, L=None, N=None):
        """Return Qx."""
        x = super().check_x(x)
        Qx_ = np.zeros_like(x)
        Qx_[x < L] = Q - N * x[x < L]
        Qx_ = WellBase.itimize(Qx_)
        return Qx_
    
    def h(self, Q=None, x=None, L=None, N=None):
        """Return head using unconfined k and D."""
        WellBase.check_keys({'k', 'D'}, self.aq)
        x = np.atleast_1d(x).astype(float)
        Hinf = self.aq['D']
        h = np.zeros_like(x) + Hinf
        h[x < L] = np.sqrt(Hinf ** 2  + N / self.aq['k'] * (L ** 2 - x[x < L] ** 2) -
                    2 * Q / self.aq['k'] * (L - x[x < L]))
        h = WellBase.itimize(h)
        return h
    
    def xdiv(self, Q=None, L=None, N=None):
        """Return location of water divide."""
        x = - Q / N
        if x > L:
            return np.nan
        else:
            return Q / N
        
class Blom1D(StripBase):
    """Class for Blom's solutions for 1D flow.
    
    Blom has Verruijt for x < x[dd == Nc] and mazure for x > x[dd == Nc]
    """
    def __init__(self, aqprops={}):
        super().__init__(aqprops=aqprops)
        
        WellBase.check_keys({'kD', 'c'}, self.aq)
        return

    def getL(self, Q=None, N=None):
        """Return L where drawdown is Nc."""
        L = Q / N - self.aq['lambda']
        L = L if L > 0. else 0.
        return L
        
    def h(self, Q=None, x=None, N=None):
        """Return h using for unconfined aquifer 'k' and 'D' as Hinf."""
        WellBase.check_keys({'k', 'D', 'c'}, self.aq)        
        Hinf = self.aq['D']
        HL = self.aq['D'] - N * self.aq['c']
        L = self.getL(Q=Q, N=N)
        h2 = HL ** 2 + N / self.aq['k'] * (L ** 2 - x ** 2) - 2 * Q0 / (self.aq['k']) * (L - x)
        if np.isscalar(x):
            if x > L:
                return Hinf - N * self.aq['c'] * np.exp(-(x - L) / self.aq['lambda'])
            else:
                return np.sqrt(h2)
        h = np.zeros_like(x)
        h[x < L]  = np.sqrt(h2[x < L])
        h[x >= L] = Hinf - N * self.aq['c'] * np.exp(-(x[x >= L] - L) / self.aq['lambda'])
        h = WellBase.itimize(h)
        return h

    def dd(self, Q=None, x=None, N=None):
        """Return drawdown using uniform 'kD' of aquifer."""
        x = super().check_x(x)
        L = self.getL(Q=Q, N=N)
        s = np.zeros_like(x)
        s[x < L] = Q / self.aq['kD'] * ((L - x[x < L]) -
                N / (2 * self.aq['kD']) * (L **2 - x[x < L] ** 2) + N * self.aq['c'])
        s[x >= L] = N * self.aq['c'] * np.exp(-(x[x >= L] - L) / self.aq['lambda'])
        s = WellBase.itimize(s)
        return s
        
    def Qx(self, Q=None, x=None, N=None):
        """Return Q(x)."""
        x = super().check_x(x)
        L = self.getL(Q=Q, N=N)
        Qx_ = np.zeros_like(x)
        Qx_[x < L] = Q - N * x[x < L]
        Qx_[x >= L] = N * self.aq['lambda'] * np.exp(- (x[x >= L] - L) / self.aq['lambda'])
        Qx_ = WellBase.itimize(Qx_)
        return Qx_
        
    def xdiv(self, Q=None, N=None):
        """Return x location of water divide."""
        L = self.getL(Q=Q, N=N)
        if Q / N < L:
            return L
        else:
            return np.nan
        
        
# %%

# %% [markdown]
# # Formule van Verruijt versus die van Dupuit en de formule van Blom
# 
# ## Het idee achter deze formules
# 
# Het idee achter de formule van Verruijt is een onttrekking in het centrum van een circulair gebied dat gevoed wordt met een constant en uniform neerslagoverschot. Het stroombeeld is hiermee uniek bepaald. Het stijghoogtebeeld is echter alleen bepaald wanneer deze ergens, op een, op zichzelf willekeurige afstand van de put wordt gefixeeerd. De formule van Verruijt is stationair and kan direct worden afgeleid voor een watervoerend pakket met vrije watertafel, waarin de effectieve dikte van het watervoerend pakket varieert met de hoogte van de watertafel. Als het gebied groot genoeg is, is er altijd een afstand waarbinnen de voeding van de put geheel richting put stroomt, en waarbuiten dit van de put af stroomt. Deze afstand vormt de waterscheiding, waarop de watertafel horizontaal is. De afstand tot deze waterscheiding neemt toe met de grootte van de onttrekking.
# 
# De wiskundige oplossing voor de situatie die Verruijt voor ogen had, is direct  vergelijkbaar met die volgens Dupuit, die geen rekening hield met het neerslagoverschot. De verlaging volgens Verruijt en Dupuit, dat wil zeggen de oorspronkelijke stijghoogte minus de nieuwe, verlaagde stijghoogte, is voor beide situaties dezelfde.
# 
# De situatie die Blom voor ogen had, is een put in heen gebied met voldoende sloten, zodat de weerstand de sloten mag worden beschouwd als een vlakdekkende drainageweerstand. Voorafgaand aan de onttrekking draineren deze sloten de voeding van het gebied vanuit het neerslagoverschot. Binnen een nader te bepalen afstand rond de put vallen alle sloten droog; daarbuiten blijven zijn draineren. De drainage door de sloten die droogvallen komt volledig ten goede aan het door de put onttrokken water. Buiten het gebied met de nu droge sloten is de drainage afgenomen maar niet tot nul, en komt zodoende alleen de afgenomen drainage ten goede aan het onttrokken water. De drainage op de grens van het gebied met droge sloten is juist gelijk aan nul, daarbuiten neemt de deze asymptotisch toe tot het totale neerslagoverschot.
# 
# Bij een vlakdekkende drainageweerstand $c$, per definitie gelijk aan de gemiddelde grondwaterstand boven het slootpeil, $h$ gedeeld door het neerslagoverschot $N$, is de verlaging $s_R$ op de grens van het gebide met drooggevallen sloten en niet-drooggevallen sloten gelijk aan $s_R=Nc$.
# 

# %% [markdown]
# ## Verruijt voor 1D stroming alleen langs de x-as
# 
# We kunnen de Formule van Verruijt zowel afleiden voor radiale situatie als die met uitsluitend stroming in de $x$-richting. Beide formules kunnen in de praktijk van pas komen en onderlinge vergelijking geeft een beter inzicht in het gedrag dat ze beschrijven.
# 
# In de eendimensionale situatie is de onttrekking geen put maar ene lijnonttrekking en is de cirkelvormige rand een rechte rand met gegeven grondwaterstand op een gegeven afstand $L$. De lijnonttrekking $Q$ heeft dimensie [L2/T] in plaats van [L2/T] bij de axiaal symmetrische situatie.
# 
# De waterbalans op afstand $x$ van de onttrekking bij neerslagoverschot $N$ is dan
# 
# $$ Q(x) = Q(0) - N x = + kh \frac{dh}{dx}$$
# 
# waarbij de stroming naar de onttrekking toe (links) posiftief wordt genomen tegen de richting $x$ in, zodat een onttrekking postief is.
# 
# $$ Q_0 x - \frac 1 2 N x^2 = \frac 1 2 k h^2 + C$$
# 
# en met gegeven stijghoogte $H$ op $x=L$ volgt voor $C$
# 
# $$ Q_0 L - \frac 1 2 N L^2 = \frac 1 2 k H^2 + C$$
# 
# Deze twee vergelijkingen van ekaar aftrekken elimineert constante $C$, zodat
# 
# $$ Q_0 (x - L) - \frac 1 2 N (x^2 - L^2) = \frac 1 2 k (h^2 - H^2) $$
# 
# oftewel
# 
# $$ h^2 - H^2  = \frac N k (L^2 - x^2) - \frac {2 Q_0} k (L - x)$$
# 
# #### Waterscheiding
# 
# $$ 2 h \frac{dh}{dx} |_{x=x_D} = - 2 \frac N k x_D + 2 \frac{Q_0} k = 0. \,\,\,\,\,\,\rightarrow\,\,\,\,\,\, x_D =\frac {Q_0} N$$
# 
# 
# 
# Bij constant doorlaatvermogen kunnen we $h + H$ afsplitsen en schrijven als $2D$ met $D$ de pakketdikte
# 
# $$ h^2 - H^2 = (h - H)(h + H) = (h - H) 2D $$
# 
# Zodat in bij uniform doorlaatvermogen $kD$ geldt
# 
# $$ h - H  = \frac N {2kD} (L^2 - x^2) - \frac {Q_0} {kD} (L - x)$$
# 
# Met $h-H = -s$ de verlaging, volgt voor deze verlaging de volgende uitdrukking
# 
# $$ s  = \frac {Q_0} {kD} (L - x) -\frac N {2kD} (L^2 - x^2)$$

# %% [markdown]
# ## Formule van Blom, ééndimensionaal
# 
# Bij blom is de stijghoogte op $x=L$ niet gefixeerd, maar is de drainage daar net nul. Er is daar dus nog een restverlaging die wordt veroorzaakt door de lek (afvoerreductie) die de onttrekking veroorzaakt in het gebied waar de sloten nog wel blijven drainenen, zij het niet het volledige neerslagoverschot. Schrijven we de grondwaterstand op $x=\infty$ gelijk aan $H_\infty$ en die op $x=L$ als $H_L$ dan volgt met $h^2 - H_L^2 = (h^2 - H_\infty^2) - (H_L^2 - H_\infty^2)$
# 
# 

# %% [markdown]
# ### Bij variabele pakketdikte
# 
# $$ h^2 - H_\infty^2  = \frac N k (L^2 - x^2) - \frac {2 Q_0} k (L - x) + (H_R^2 - H_\infty^2)$$
# 
# De afgeleide naar $x$ is
# 
# $$ 2 h_L \frac{dh}{dx} = - 2 \frac {N L} k + \frac {2 Q_0} k$$
# 
# $$ \frac{dh}{dx} = - \frac {N x} {k h} + \frac {Q_0} {k h}$$
# 
# en op $x=L$ is dit
# 
# $$ \frac{dh}{dx}|_{x=L} = - \frac {N L} {k h_L} + \frac {Q_0} {k h_L}$$

# %% [markdown]
# ### Bij vaste pakkedite $D$ 
# 
# Hier geldt $h^2 - H_\infty^2 = -s\,2D$ en $H_R^2 - H_\infty^2 = -s_L\,2D$
# 
# Zodat in dat geval
# 
# $$ h - H_\infty  = \frac N {2kD} (L^2 - x^2) - \frac {Q_0} {kD} (L - x) + (H_R - H_\infty)$$
# 
# $$ -s = \frac N {2kD} (L^2 - x^2) - \frac {Q_0} {kD} (L - x) - s_L$$
# 
# en tenslotte
# 
# $$ s = \frac {Q_0} {kD} (L - x) - \frac N {2kD} (L^2 - x^2) + s_L$$
# 
# Met als afgeleide
# 
# $$ \frac{ds}{dx} = -\frac {Q_0} {kD} + \frac {Nx} {kD}$$
# 
# en voor $x=L$
# 
# $$ \frac{ds}{dx}|_{x=L} = -\frac {Q_0} {kD} + \frac {NL} {kD}$$
# 
# Behalve het teken is de wiskundige vorm van de afgeleide hetzelfde voor de situatie met als voor die zonder vaste pakketdikte. Bij variabele pakketdikte moet in plaats van de gehele pakketdikte $D$ die ter plekke van $x=L$ moet worden gebruikt $h_L$ worden ingevuld. In veel situaties is het verschil klein. Wanneer dit verschil relevant is, moet het verhang in de situatie met variabele pakketdikte iteratief worden berekend.

# %% [markdown]
# ### Voor $x > L$, waar voeding door lek (verminderde afvoer) vanuit de sloten optreedt.
# 
# Voor $x > L$  hebben we voeding door lek vanuit de sloten via de drainageweerstand, die wordt gekarakteriseerd door de spreidingslengte $\lambda=\sqrt{kD c}$ met $c$ de drainageweerstand. De drainageweerstand is per definitie gelijk aan de gemiddelde grondwaterstand in het gebied met de sloten gedeeld doo de voeding, dus $c = (h_{gemiddeld} - H_{sloot}) / N$, met dimensie [L].
# 
# De oplossing voor de eendimensionale stationaire stroming in het gebied met nog drainerende sloten is dan
# 
# $$ s_{x>L} = s_L \exp(-\frac {x - L} \lambda),\,\,\,\,\,\, \lambda = \sqrt{kD c} $$
# 
# en de stroming
# 
# $$Q_{x \ge L} = s_L \frac{kD}{\lambda} \exp \left(-\frac {x-L} \lambda \right) $$

# %% [markdown]
# Onttrekking Q_0 [m2/d] op $x=0$, met voeding uit neerslag gelijk aan N, waarbij de sloten tot $x=L$ droogvallen en daarbuiten bijven draineren. Waar voor de onttrekking alle sloten de voeding wegdraineerden, is dat voor $x < L$ niet meer het geval. Deze voeding komt nu ten goede aan de onttrekking. De situatie is dan hetzelfde als in het eendimensionale geval van Verruijt.
# 
#  Bij constante aangenomen doorlaatvermogen $kD$ geeft dit de volgende verlaging
# 
# $$ s = \frac{Q_0 (L - x)}{kD} - \frac{N \left(L^2 - x^2\right)}{2 kD} + s_L$$
# 
# met $s_L$ de verlaging op $x=L$
# 
# De afgeleide op $x=L$
# 
# $$ \frac{ds}{dx}|_{x=L} = -\frac{Q_0 L}{kD} + \frac{NL}{kD}$$
# 
# 
# Voor $x > L$ wordt het pakket uitsluitend gevoed door lek, die wordt gekarakteriseerd door de spreidingslengte
# 
# $$\lambda = \sqrt{kD c}$$
# 
# waarin $c$ de zogenoemde drainageweerstand. Dit is de gemiddelde grondwaterstand tussen de sloten gedeeld door het het gedraigneeerde neerslagoverschot. De verlaging voor $x>L$ is dan
# 
# $$s_x = s_L \exp \left(- \frac {x-L} \lambda \right)$$ 
# 
# En het debiet $Q_{x \ge L} = -kD \frac{ds}{dx}$. Merk op dat $Q$ steeds positief is genomen in de negatieve $x$-richting, dus wanneer de stijghoogte met $x$ toeneemt en de verlaging met $x$ afneemt
# 
# $$ Q_{x \ge L} = -s_L \frac{kD} \lambda \exp \left(- \frac {x-L} \lambda \right)$$
# 
# end dus met voor $x=L$, waar $Q_L = Q_0 - L N$
# 
# $$ Q_0 - L N =  -s_L \frac{kD} \lambda$$
# 
# We kennen ook de verlaging $s_L$ voor $x=L$, want daar is de voeding gelijk aan de lek via de drainageweerstand
# 
# $$x=L \,\,\,\,\,\,\rightarrow\,\,\,\,\,\, N=\frac{H_\infty - H_L}{c} = \frac {s_L}{c}\,\,\,\,\,\,\rightarrow\,\,\,\,\,\, s_L = N c$$
# 
# met $ N c kD / \lambda = N c \lambda$ volgt
# 
# $$ Q_{x \ge L} = -N \lambda \exp \left(- \frac {x-L} \lambda \right)$$
# 
# en dus hebben wel voor $x=L$
# 
# $$Q_0 - L N = N c \frac{kD} \lambda$$
# 
# $$Q_0 - L N = N \lambda$$
# 
# en tenslotte
# 
# $$L = \frac{Q_0}{N} - \lambda$$
# 
# Voor de situatie met variabele dikte kan een correctie hierop worden uitgevoerd
# 
# $$L = \frac{Q_0}{N} - \frac {kh_Lc} \lambda =  \frac{Q_0}{N} - \frac{k h_L c}{\sqrt{kDc}} = \frac{Q_0}{N} - \lambda \sqrt{\frac {h_L} D} $$

# %% [markdown]
# 
# #### Voor drainageweerstand $c=0$, gaat de formule van Blom over in die van Verruijt
# 
# Voor $c=0$ verloopt de voeding vanuit de sloten zonder enige weerstand. De plossing is dan dezelfde als met een vaste rand op $x=L$. Met toenemende weerstand en of doorlaatvermogen neemt $L$ af. $L=0$ wanneer $\frac{Q_0} N = \lambda$ dus wanneer
# 
# $$Q_L = Q_0 = N \lambda$$
# 
# Dit is de situatie wanneer de verlaging op $x=0$ exact gelijk is aan $N c$ en de sloot op $x=0$ dus net niet meer draineert.
# 
# En wanneer $L=0$ geldt in feite dat de verlaging op $x=0$ kleiner of gelijk is aan $N c$ en de sloot op $x=0$ nog wel (enigszins) draineert.
# 
# $$ s_{x>0} = s_0 \exp \left(-\frac x \lambda \right)\,\,\,\,\,\, met\,\,\,\,\,\, Q_0 = s_0 \frac {kD} \lambda
# \le  N c \frac{kD} \lambda = N \frac {\lambda^2} \lambda = N \lambda$$
# 
# en daar hier $s_0 = N c$ volgt
# 
# $$ Q_0 = N c \frac {kD} \lambda  = N \lambda$$
# 
# Dus, $L=0$ wanneer de onttrekking zodanig is dat de verlaging op $x=0$ precies gelijk is aan de voeding.
# 
# Wellicht is interessant op te merken, dat de totale toestroming vanuit het gebied met nog wel drainerende sloten, $r \ge R$ gelijk is aan $N \lambda$ dus als het ware geschiedt vanuit een strook sloten ter breedte $\lambda$.
# 
# In de situatie met sloten die deels droogvallen kan de afstand $L$ tot waar dat het geval is direct worden berekend uit
# 
# $$L = \frac{Q_0}{N} - \lambda$$
# 
# Zoals we zullen zien is dit is bij axiaal symmetrische stroming niet het geval.

# %% [markdown]
# ## Verruijt axiaal symmetrisch

# %% [markdown]
# ### Variabele pakketdikte $h$
# 
# De formule van Verruijt voor axiale stroming naar een put gaat er ook van uit dat de onttrekking geheel wordt gevoed vanuit het neerslagoverschot. Ook hier is de stroming volledig bepaald door de onttrekking en het neerslagoverschot
# 
# $$ Q_r = Q_0 - \pi r^2 N = 2 \pi r k h \frac{dh}{dr} = \pi r k \frac {dh^2}{dr} $$
# 
# oftewel
# 
# $$ \frac{dh^2}{dr} = \frac {Q_0}{\pi k r} - \frac {r N }{k}$$
# 
# Integratie levert
# 
# $$ h^2 = \frac{Q_0}{\pi k} \ln r - \frac N {2 k} r^2 + C $$
# 
# De constante invullen geeft
# 
# $$ h_R^2 = \frac{Q_0}{\pi k} \ln R - \frac N {2 k} R^2 + C $$
# 
# Beide vergelijkingen van elkaar aftrekken elimineert deze integratieconstante
# 
# $$ h^2 - h_R^2 =  -\frac{Q_0}{\pi k} \ln \frac R r + \frac N {2 k} \left(R^2 - r^2\right) $$
# 
# de afgeleide van $dh/dr$ is
# 
# $$ \frac{dh}{dr}=\frac{Q_0}{2 \pi k h} \frac 1 r - \frac{N r}{ 2 k h} $$
# 
# In deze uitdrukking komt de nog onbekende pakketikte $h$ in het rechter lid voor, maar valt eruit op de waterscheiding, $r=R_S$, wanneer we de afgeleide gelijk aan nul stellen. dit levert
# 
# $$ \frac{Q_0}{2 \pi k h} \frac 1 R_S = \frac{N R_S}{k h} $$
# 
# Oftewel
# 
# $$ R_S = \sqrt{\frac{Q_0}{\pi N}} $$

# %% [markdown]
# ### Constante pakkedikte $D$
# 
# En voor een constante pakketdikte $D$, met $h^2 - h_R^2 = (h - h_R) \, 2 D$ en $h-h_R = -s$, wordt de verlaging 
# 
# $$ h - h_R = -s =-\frac{Q_0}{2 \pi kD} \ln \frac R r + \frac N {4 kD} \left(R^2 - r^2\right) $$
# 
# En dus geldt
# 
# $$ s = \frac{Q_0}{2 \pi kD} \ln \frac R r - \frac N {4 kD} \left(R^2 - r^2\right) $$
# 
# De afgeleide $ds/dr$ is
# 
# $$ \frac {ds}{dr} = -\frac{Q_0}{2 \pi kD} \frac 1 r + \frac {N r}{2 kD} $$
# 
# Het teken van de twee termen in het rechter lid zijn dan precies omgekeerd. De afgeleide gelijk aan nul stellen, voor de waterbalans, voor $r=R_S$ levert nu
# 
# $$ Q_0 = \pi R_S^2 N $$
# 
# Of
# 
# $$ R_S = \sqrt{\frac{Q_0}{\pi N}}$$
# 
# Dit volgde natuurlijk al direct uit de waterbalans. Dit is ookk de reden dat de ligging van de waterscheiding voor variabele en constante pakketdikte dezelfde is.
# 

# %% [markdown]
# ## BLom axiaal-symmetrisch
# 
# De situatie die Verruijt voor ogen heeft is dezelfde als die van Blom met uitzondering van wat er gebeurt buiten radius $R$. Bij Verruijt is radius $R$ een vaste rand met verlaging nul en waarbinnen de situatie kan worden opgevat als het gebied met drooggevallen sloten waarbinnen het neerslagoverschot niet meer wordt gedraineerd, maar richting de put stroomt. Bij Blom markeert de radius $R$ ook het gebied zonder sloten, maar stroomt er ook buiten deze radius nog water in de richting van de put. De grondwaterstroom $Q_R$ is zowel bij Verruijt als bij Blom  gelijk aan $Q_0 - \pi R^2 N$. De grondwaterstroming over deze rand is bij Verruijt afkomstig van de vaste rand op $r=R$ en bij Blom komt die van grotere afstanden uit verminderde slootafvoer. Er is bij Blom op $r>R$ dus nog steeds verlaging in tegenstelling to bij Verruijt. Om dezelfde reden als hiervoor is de verlaging op $r=R$ gelijk aan $s_R = N c $.

# %% [markdown]
# ### Verlaging voor $r \le R$
# 
# De verlaging voor $r \le R$ is dezelfde als die bij Verruijt met dien verstande dat de velaging op $r=R$ niet nul is maar $s_R = N c $
# 
# 

# %% [markdown]
# ### Verlaging voor $r \ge R$ (De Glee (1930))
# 
# Bij Blom vallen de sloten droog binnen een straal gelijk aan $R$. De stroom $Q_R$ op $r=R$ volgt uit de waterbalans en is gelijk aan $Q_r = Q_0 - \pi R^2 N$. Het gebied voor $r>R$ fungeert als een semi-gespannen aquifer, die gevoed wordt uit verminderde drainage door de sloten, die gelijk is aan de verlaging ter plekke gedeeld door de drainageweerstand. De wiskundige oplossing voor de grondwaterstromign in deze situatie is volgens De Glee (1930); zij bestaat alleen voor constante pakketdikte:
# 
# $$ s_{r\ge R} = \frac{Q_R}{2 \pi kD}\cdot \frac{K_0 \left(\frac r \lambda\right)}{\frac R \lambda K_1\left(\frac R \lambda\right)} $$
# 
# $$ Q(r) = Q_R \cdot \frac r R \cdot \frac{K_1\left(\frac r \lambda\right)}{K_1\left(\frac R \lambda\right)}$$
# 
# $Q_R$ volgt direct uit de waterbalans over het gebied gegeven door $r\le R$
# 
# $$ Q_R = Q_0 - \pi R^2 N$$
# 
# De verlaging op $r=R$ is gelijk aan $s_R$
# 
# $$ s_R = \frac{Q_R}{2 \pi kD}\cdot \frac{K_0 \left(\frac R \lambda\right)}{\frac R \lambda K_1\left(\frac R \lambda\right)} $$
# 
# De verlaging op $r=R$ is ook gelijk aan $s_R = N c$ omdat daar de verminderde drainage juist de resterende drainage nul maakt. Hiermee hebben we een vergelijking waarmee de radius kan worden berekend waarbinnen de sloten droog vallen
# 
# $$ N c = \frac{Q_R}{2 \pi kD}\cdot \frac{K_0 \left(\frac R \lambda\right)}{\frac R \lambda K_1\left(\frac R \lambda\right)} $$
# 
# We kunnen $R$ alleen iteratief berekenen, bijvoobeeld met de Newton methode. Deze methode maakt van bovenstaande expressie een functie van $y(R)$ waarvan $R$ het nulpunt is dat moet worden gevonden.
# 
# $$ y(R) = - N c + \frac{Q_R}{2 \pi kD}\cdot \frac{K_0 \left(\frac R \lambda\right)}{\frac R \lambda K_1\left(\frac R \lambda\right)} $$
# 
# De Newton methode vindt het nulpunt iteratief als volgt
# 
# $$R_{n+1} = R_N - \frac{y(R)}{y \prime{R}}$$
# 
# waarin $y \prime(R)$ de afgeleide is van $y(R)$ naar $R$ en $n$ de betreffende iteratie.
# 
# Deze methode is eenvoudig, maar vergt wel de afgeleide van $y(R)$. Deze kan zowel numeriek met $y(R)$ worden berekend als anlytisch. Voor dit laatste moeten we $y(R)$ differentiëren naar $R$.
# 
# Voor de afgeleide van $y$ naar $R$ hebben we de afgeleide van de factor met de besselfuncties nodig
# 
# Gebruik makend van $\frac{d K_0 z}{dz} = -K_1 z$ en $\frac{d (z K_1 z)}{dz} = -z K_0 z$ volgt na enige uitwerking
# 
# $$\frac{d}{dr}\left(\frac{K_0\left(\frac R \lambda\right)}{\frac R \lambda K_1\left(\frac R \lambda\right)}\right) 
# = \frac 1 R \left( \frac{K_0^2 \left(\frac R \lambda\right)}{K_1^2\left(\frac R \lambda\right)} - 1\right) 
# $$
# 

# %% [markdown]
# 
# waarmee uiteindelijk na invoegen in de gehele uitdrukking voor $dy(R)/dr$
# 
# $$\frac{dy(R)}{dr} = -\frac{\lambda N}{kD} \cdot \frac{K_0\left(\frac R \lambda\right)}{K_1\left(\frac R \lambda\right)} -
# \left(\frac{Q_0 / R}{2 \pi kD} - \frac{N R} {2 kD} \right)
# \cdot\left(
#     1 -\frac{K_0^2\left(\frac R \lambda\right)}{K_1^2\left(\frac R \lambda\right)}
# \right)$$
# 
# Deze functie $y(R)$ en $dy(R)/dR=y'(R)$ worden hierna geïmplementeerd en meegenomen in de Newton methoden om $R$ iteratief te bepalen. We zullen daarbij de analytische bepaalde afgeleide controleren met de numeriek bepaalde afgeleide.
# 
# Wanneer we $R$ kennen, kunnen we zowel de verlaging voor $r\le R$ als die voor $r\ge R$ berekenen.

# %% [markdown]
# ## Implementatie Verruijt en Blom eendimensionaal
# 
# Voor 1D berekening van de verlaging of de stijghoogte bij Verruijt moeten we afstand $L$ opgeven waar de verlaging nu is. Voor BLom is dit de afstand waarop de verlaging gelijk is aan $s_L = N c$. Bij Blom kunnen we die bij gegeven onttrekking en neerslagoverschot berekenen mits $kD$ en drainageweerstand $c$ bekend zijn.

# %% [markdown]
# ## Voorbeeld Verruijt 1D, vaste Q0, variërende L

# ================= E X A M P L E S ===================================
if __name__ == '__main__':
    # %%
    # Verruijt 1D fixed Q0 and varying L

    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 0.5, 'N': 0.003, 'L': 250}
    x = np.linspace(0, 300, 301)[1:]

    V1 = Verruijt1D(aqprops)
    h = V1.h(x=x, **pars)
    # print(h)

    # Head
    ax = newfig(f"Verruijt 1D, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}", "x [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for L in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)
        pars['L'] = L
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = V1.h(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        hd = V1.h(x=xd, **pars)
        ax.plot([-xd, xd], [hd, hd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [aqprops['D'], aqprops['D']], '.',
                color=clr, label=f'L={pars['L']}')

    ax.legend(fontsize=6, loc='lower right')

    
    # Drawdown
    ax = newfig(f"Verruijt 1D, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}", "x [m]", "drawdown [m]", figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for L in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)        
        dd = V1.dd(x=x, **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), V1.dd(x=x, **pars), color=clr, label=label)

        xd = V1.xdiv(**pars)
        ddnd = V1.dd(x=xd, **pars)
        ax.plot([-xd, xd], [ddnd, ddnd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [0., 0.], '.', color=clr, label=f'L={pars['L']}')

    ax.legend(fontsize=6, loc='lower right')

    #plt.show()


    # %% [markdown]
    # ## Voorbeeld Verruijt 1D; vaste L, variërende Q0
    # 
    # De onttrekking en, daarmee de verlaging is stapsgewijs vergroot. Hierdoor komt de waterscheiding steeds verder weg te liggen tot deze de vaste rand $L$ bereikt.
    # 
    # Bij verruijt is buiten de vaste rand geen stroming en dus ook geen verlaging.
    # 
    # Bij grote onttrekking is de daling van de grondwaterstand in het eeste plaatje is een stuk groter dan de verlaging in het tweede plaatje. Dit is het gevolg van de afname van de dikte van het watervoerende pakket waar in het eerste plaatje wel en in het tweede plaatje geen rekening mee is gehouden.

    # %%
    # Verruijt 1D, fixed L and varying Q0

    pars['L'] = 600
    x = np.linspace(0, 1000, 5001)[1:]

    # head
    ax = newfig(f"Blom 1D, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}", "x [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for Q in [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        clr = next(clrs)
        pars['Q'] = Q
        label=f"Q0 = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = V1.h(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        hd = V1.h(x=xd, **pars)
        ax.plot([-xd, xd], [hd, hd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [aqprops['D'], aqprops['D']], '.', color=clr, label=f'L={pars['L']}')

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    # plt.show()

    # Drawdown
    ax = newfig(f"Verruijt 1D, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}", "x [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q0 in [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        clr = next(clrs)
        pars['Q'] = Q0
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        dd = V1.dd(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        ddnd = V1.dd(x=xd, **pars)
        ax.plot([-xd, xd], [ddnd, ddnd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [0., 0.], '.', color=clr, label=f'L={pars['L']}')

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    #plt.show()

    
    # %% [markdown]
    # ## Voorbeeld Blom 1D, variërende Q0
    # 
    # Bij Blom wordt de $L$ berekend, zodanig dat de verlaging op $x=L$ gelijk is aan $N c$.
    # 
    # In dit voorbeeld worden de grondwatertstand en de verlaging berekend voor verschillende waarden van $Q0$. Voor elke Q0 wordt de afstand $L$ berekend waarbinnen de sloten droogvallen. Op deze afstand is de verlaging gelijk aan $N c$, waardoor op afstand $L$ de sloot juist droogvalt (of beter: de sloot daar net niet meer draineert).
    # 
    # We zien verder dat de verlaging in het eerste plaatje, met variabele pakketdikte voor gotere onttrekkingen groter is dan die in het tweede plaatje, voor de situatie met vaste pakketdikte.
    # 
    # Voor de situatie met variabele pakketdikte zou de stijghoogte op $x > L x$ nog iets gecorrigeerd kunnen worden voor de in werkelijkheid afnemende dikte. Dit effect is echter zo klein dat het verschil in de aansluiting op het aangegeven punt, dus $x=L$ in de grafiek niet is te zien. Deze correctie kan eenvoudig worden verwaarloosd in de praktijk.In het onderhavige geval is dit een correctie van de $\lambda van $h/H = (D - s_L) / D \approx 19.5 / 20 \approx 0.98$ op de gebruikte waarde van $\lambda$, dus verwaarloosbaar.

    # %%
    # Blom 1D, fixed L and varying Q0

    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 0.5, 'N': 0.003}
    x = np.linspace(0, 300., 301)[1:]

    B1 = Blom1D(aqprops)
    h = B1.h(x=x, **pars)

    B1.dd(x=xd, **pars)
    x = np.linspace(0, 1000, 501)[1:]

    # head
    ax = newfig(f"Blom 1D, stijghoogte, k={aqprops['k']:.1f} m/d, H={aqprops['D']:.1f} m, Nc={pars['N'] * aqprops['c']:.3g} m, lambda={aqprops['lambda']:.3g} m",
                "x [m]", "drawdown [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for Q0 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        clr = next(clrs)
        pars['Q'] = Q0
        h = B1.h
        L = B1.getL(**pars)
        label=f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = B1.h(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = B1.xdiv(**pars)
        hd = B1.h(x=xd, **pars)
        hL = B1.h(x=pars['L'], **pars)    
        if not np.isnan(xd):
            ax.plot([-xd, xd], [hd, hd], 'o', color=clr, label=f"xd={xd:.3g} m, hd={hd:.3g} m")
        
        ax.plot([-pars['L'], pars['L']], [hL, hL], '.', color=clr, label=f'L={pars['L']:.3g} m, hL={hL:.3g} m')

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    #plt.show()

    # Drawdown
    ax = newfig(f"Blom 1D, verlaging, k={aqprops['k']:.1f} m/d, D={aqprops['D']:.1f} m, Nc={pars['N'] * aqprops['c']:.3g} m, lambda={aqprops['lambda']:.3g} m",
                "x [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q0 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        clr = next(clrs)
        pars['Q'] = Q0
        L = B1.getL(**pars)
        label = label=f"Q = {pars['Q']:.3g} m2/d, N={pars['N']:.3g} m/d"
        dd = B1.dd(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        xd = B1.xdiv(**pars)
        ddnd = B1.dd(x=xd, **pars)    
        ddnL = B1.dd(x=L, **pars)
        if not np.isnan(xd):
            ax.plot([-xd, xd], [ddnd, ddnd], 'o', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [ddnL, ddnL], '.', color=clr, label=f'L={pars['L']:.3g} m, dd={ddnL:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    plt.show()
    
    sys.exit()

    # %% [markdown]
    # ## Voorbeelden Verruijt, axiaal symmetrisch

    # %%
    # Verruijt 1D fixed Q0 and varying R

    aqprops = {'k': 10., 'D': 20., 'c': 200., 'R':250.}
    pars = {'Q0': 300, 'N': 0.003}
    r = np.linspace(0, 300, 301)[1:]

    V2 = Verruijt(**aqprops)
    h = V2.h(r=r, **pars)
    # print(h)

    # Head
    ax = newfig(f"Verruijt axiaal-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for R in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)
        V2.R = R
        label = f"Q0 = {pars['Q0']:.3g}, N={pars['N']:.3g}"
        h = V2.h(r=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr, label=label)

        rd = V2.rdivh(**pars)
        hd = V2.h(r=rd, **pars)
        ax.plot([-rd, rd], [hd, hd], 'v', color=clr, label=f"rd={rd:.3g} m, hd={hd:.3g} m")
        ax.plot([-V2.R, V2.R], [V2.D, V2.D], '.', color=clr, label=f'R={R:.3g} m, h={V2.D:.3g} m')

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    plt.show()

    # Drawdown
    ax = newfig(f"Verruijt axial-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for R in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)
        V2.R = R
        dd = V2.dd(r=r, **pars)
        label = f"Q0 = {pars['Q0']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        rd = V2.rdivD(**pars)
        ddnd = V2.dd(r=rd, **pars)
        ax.plot([-rd, rd], [ddnd, ddnd], 'v', color=clr, label=f"rd={rd:.3g} m, ddnd={ddnd:.3g} m")
        ax.plot([-V2.R, V2.R], [0., 0.], '.', color=clr, label=f'R={R:.3g} m, ddnd={0.:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    plt.show()


    # %%
    # Bodemconstanten, gebiedsradius, voeding, onttrekking en putstraal
    aqprops = {'k': 20, 'D': 20, 'R':1000.}
    pars = {'Q0': 1200., 'N': 0.002}

    V2 = Verruijt(**aqprops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Verruijt k={k:.0f} m/d, H=vast={aqprops['D']:.0f} m, N={pars['N']:.3g} m/d, R={aqprops['R']:.0f} m")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("h [m]"),
                
    # Kleuren en debieten
    clrs = cycle('brgmck')
    Qs =[0.001, 500, 1000, 2000, 3000]

    r = np.logspace(0, 3, 31)
    R = aqprops['R']
    H = aqprops['D']

    for Q0 in Qs:
        pars['Q0'] = Q0
        clr = next(clrs)    
        # Stijghoogte (links en rechts)
        dd = V2.dd(r=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), H - np.hstack((dd[::-1], dd)), color=clr,
                label=f'Q0={Q0:.0f} m3/d')

        # Intrekgebied, radius = rI, links en rechts    
        rI = V2.rdivD(**pars)
        ddnI = V2.dd(r=rI, **pars)
        ax.plot([-rI, rI], [H - ddnI, H - ddnI], ls='none', marker='v', mfc=clr,
                mec='k', ms=6, label=f'r_intr.={rI:.0f} m, dd={ddnI:.3g} m', zorder=5)    

    # Vaste randen
    H = aqprops['D']
    ax.plot([+R, +R], [0, H], '--', color='blue', lw=2, label='vaste rand')
    ax.plot([-R, -R], [0, H], '--', color='blue', lw=2, label='')

    # Put
    rw = 0.5
    ax.plot([+rw, +rw], [0, H], ':', color='blue', lw=2, label='put')
    ax.plot([-rw, -rw], [0, H], ':', color='blue', lw=2, label='')

    # Pakket bodem
    ax.add_patch(patches.Rectangle((-R, -2), 2 * R, 2, fc='gray'))

    ax.text(0.6, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    leg = ax.legend(loc='lower left', fontsize=10)
    leg.set_bbox_to_anchor((0.10, 0.11, 0.3, 0.5), transform=ax.transAxes)

    # %%
    # Bodemconstanten, gebiedsradius, voeding, onttrekking en putstraal
    aqprops = {'k': 10, 'D': 20, 'R':1000.}
    pars = {'Q0': 1200., 'N': 0.002}

    V2 = Verruijt(**aqprops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Verruijt k={k:.0f} m/d, D=variabel, HR={aqprops['D']:.0f} m, N={pars['N']:.3g} m/d, R={aqprops['R']:.0f} m")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("h [m]"),
                
    # Kleuren en debieten
    clrs = cycle('brgmck')
    Qs =[0.001, 500, 1000, 1500, 2000] #  2500]

    r = np.logspace(0, 3, 31)
    R = aqprops['R']

    for Q0 in Qs:
        pars['Q0'] = Q0
        clr = next(clrs)    
        # Stijghoogte (links en rechts)
        h = V2.h(r=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr,
                label=f'Q = {Q0:.0f}')

        # Intrekgebied, radius = rI, links en rechts
        rI = np.sqrt(Q0 / (np.pi * pars['N']))
        rI = V2.rdivh(**pars)
        hI = V2.h(r=rI, **pars)
        ax.plot([-rI, rI], [hI, hI], ls='none', marker='v', mfc=clr,
                mec='k', ms=6, label='intrekgrens', zorder=5)    

    # Vaste randen
    H = aqprops['D']
    ax.plot([+R, +R], [0, H], '--', color='blue', lw=2, label='vaste rand')
    ax.plot([-R, -R], [0, H], '--', color='blue', lw=2, label='')

    # Put
    rw = 0.5
    ax.plot([+rw, +rw], [0, H], ':', color='blue', lw=2, label='put')
    ax.plot([-rw, -rw], [0, H], ':', color='blue', lw=2, label='')

    # Pakket bodem
    ax.add_patch(patches.Rectangle((-R, -2), 2 * R, 2, fc='gray'))

    ax.text(0.6, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    leg = ax.legend(loc='lower left', fontsize=10)
    leg.set_bbox_to_anchor((0.10, 0.11, 0.3, 0.5), transform=ax.transAxes)

    # %% [markdown]
    # ## Voorbeeld Blom, axiaal symmetrisch

    # %%
    # Blom axiaal symmetrisch, fixed Q0 and varying R

    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q0': 300, 'N': 0.003}
    r = np.linspace(0, 300, 301)[1:]

    B2 = Blom(**aqprops)

    # Head
    ax = newfig(f"Blom, axiaal-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for Q0 in np.array([1., 1.5, 2., 2.5, 3.0]) * 500.:
        clr = next(clrs)
        pars['Q0'] = Q0
        R = B2.getR(R=B2.R, **pars)
        label = f"Q0 = {pars['Q0']:.3g}, N={pars['N']:.3g}"
        h = B2.h(r=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr, label=label)

        rd = B2.rdivh(**pars)
        hd = B2.h(r=rd, **pars)
        ax.plot([-rd, rd], [hd, hd], 'o', color=clr, label=f"rd={rd:.3g} m, hd={hd:.3g} m")
        HR = B2.D - pars['N'] * B2.c
        ax.plot([-R, R], [HR, HR], '.', color=clr, label=f'R={R:.3g} m, h={HR:.3g} m')

    ax.text(0.1, 0.6, "Variabele pakketdikte voor r < R", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    plt.show()

    # %%
    # Drawdown
    B2 = Blom(**aqprops)

    ax = newfig(f"Blom, axial-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q0 in np.array([1., 1.5, 2., 2.5, 3.0]) * 500.:
        clr = next(clrs)
        pars['Q0'] = Q0
        R = B2.getR(R=B2.R, **pars)    
        dd = B2.dd(r=r, **pars)
        label = f"Q0 = {pars['Q0']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        rd = B2.rdivD(**pars)
        ddnd = B2.dd(r=rd, **pars)
        ax.plot([-rd, rd], [ddnd, ddnd], 'o', color=clr, label=f"rd={rd:.3g} m, ddnd={ddnd:.3g} m")
        ddnR = pars['N'] * B2.c
        ax.plot([-R, R], [ddnR, ddnR], '.', color=clr, label=f'R={R:.3g} m, ddnd={0.:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte voor r < R", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    plt.show()

    # %% [markdown]
    # ### Demonstratie van de voortgang van het iteratieproces volgens Newton om R te vinden waar de verlaging gelijk is aan Nc
    # 
    # De afstand van de put waarop de verlaging precies gelijk is aan Nc, het critierium voor juist droogvallen van de sloten wordt iteratief berekend met de methode van Newton. Het voorschrijden van dit iteatieproces wordt hieronder grafisch weergegeven.
    # 
    # Voor de iteraties is de afgeleide nodig van de fuctie $y(R)$ zie boven. De tweede grafiek toont de afgeleide, zowel analytisch als numeriek berekent ter controle.

    # %%
    aqprops = {'k': 30.0, 'D': 20.0, 'c': 200.0, 'R': 1.0}
    pars = {'Q0': 1200., 'N':0.02}

    B2 = Blom(**aqprops)
    ax = B2.plot_newton_progress(R=1., Q0=1500.0, N=0.002)
    ax.set_ylim(-0.5, 2.5)

    ax = B2.plot_derivative_of_y(R=1, Q0=1200.0, N=0.002)

