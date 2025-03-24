# -*- coding: utf-8 -*-
"""
3D Finite Difference Models as a function.
Stream line computation (Psi) and a function
Computaion of veclocity vector for plotting with quiver

Created on Fri Sep 30 04:26:57 2016

@author: Theo

"""
import sys

sys.path.append("/Users/Theo/GRWMODELS/python/tools/")
sys.path.append("/Users/Theo/Entiteiten/Hygea/2022-AGT/jupyter")

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.special import k0 as K0, k1 as K1
import matplotlib.pylab as plt
from etc import newfig
import wellfunctionalities as wf
from fdm import mfgrid

def quivdata(Out, x, y, iz=0):
    """Returns vector data for plotting velocity vectors.

    Takes Qx from fdm3 and returns the tuple X, Y, U, V containing
    the velocity vectors in the xy plane at the center of the cells
    of the chosen layer for plotting them with matplotlib.pyplot's quiver()

    Parameters
    ----------
    'Out' dictionary of output of fdm3d
    `x` : ndarray
        grid line coordinates.
    `y` : ndarray
        grid line coordinates.
    `iz` : int
        layer number for which vectors are computed (default 0)

    Returns:
        tuple: X, Y, U,V

        X : ndaarray
            2D ndArray of x-coordinates cell centers
        Y : ndarray
            2D ndarray of y-coordinate of cell centers
        U : ndarray
            2D ndarray of x component of cell flow [L3/T]
        V : ndarray
            2D ndarray of y component of cell flow [L3/T]

    """
    Ny = len(y)-1
    Nx = len(x)-1
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])

    X, Y = np.meshgrid(xm, ym) # coordinates of cell centers

    # Flows at cell centers
    U = np.concatenate((Out.Qx[:,0,iz].reshape((Ny,1,1)), \
                        0.5 * (Out.Qx[:,:-1,iz].reshape((Ny,Nx-2,1)) +\
                               Out.Qx[:,1:,iz].reshape((Ny,Nx-2,1))), \
                        Out.Qx[:,-1,iz].reshape((Ny,1,1))), axis=1).reshape((Ny,Nx))
    V = np.concatenate((Out.Qy[0,:,iz].reshape((1,Nx,1)), \
                        0.5 * (Out.Qy[:-1,:,iz].reshape((Ny-2,Nx,1)) +\
                               Out.Qy[1:,:,iz].reshape((Ny-2,Nx,1))), \
                        Out.Qy[-1,:,iz].reshape((1,Nx,1))), axis=0).reshape((Ny,Nx))
    return X, Y, U, V


def psi(Qx, row=0):
    """Returns stream function values in z-x plane for a given grid row.

    The values are at the cell corners in an array of shape [Nz+1, Nx-1].
    The stream function can be vertically contoured using gr.Zp and gr.Xp as
    coordinates, where gr is an instance of the Grid class.

    Arguments:
    Qx --- the flow along the x-axis at the cell faces, excluding the outer
           two plains. Qx is one of the fields in the named tuple returned
           by fdm3.
    row --- The row of the cross section (default 0).

    It is assumed:
       1) that there is no flow perpendicular to that row
       2) that there is no storage within the cross section
       3) and no flow enters the model from below.
    The stream function is computed by integrating the facial flows
    from bottom to the top of the model.
    The outer grid lines, i.e. x[0] and x[-1] are excluded, as they are not in Qx
    The stream function will be zero along the bottom of the cross section.

    """
    Psi = Qx[:, row, :] # Copy the section for which the stream line is to be computed.
                        # and transpose to get the [z,x] orientation in 2D
    Psi = Psi[::-1].cumsum(axis=0)[::-1]         # cumsum from the bottom
    Psi = np.vstack((Psi, np.zeros(Psi[0,:].shape))) # add a row of zeros at the bottom
    return Psi

class Fdm3():
    """Finite difference model class."""
    # dtypes for the boundary types (alike those of flopy)    
    dtype = {
            "drn":  np.dtype([('Ig', int), ('h', float), ('C', float)]),
            "riv":  np.dtype([('Ig', int), ('h', float), ('C', float), ('rbot', float)]),
            "ghb":  np.dtype([('Ig', int), ('h', float), ('C', float)]),
            "fdr": np.dtype([('Ig', int), ('phi',   float), ('h', float),
                            ('h0', float), ('N', float), ('w', float), ('L', float)]), # For free drainage
            # phi at N, h at N, and w at N are all for q=N, h0 is ditch bottom elev. w=ditch width
        }
    
    def __init__(self, gr=None, K=None, c=None, S=None, IBOUND=None, HI=None, FQ=None):
        """Return intantiated model with grid and aquifer properties but without its boundaries.
        
        This model saves gr, K, c, S, IBOUND, HI and FQ and computes
        the system matrix without its diagonal and separately its diagonal for use in the simulation.
        This implies that IBOUND, HI and FQ may be changed later by assignment to the instantiated class.
        
        Parameters
        ----------
        gr: mfgrid.Grid object instance
            object holding the grid/mesh (see mfgrid.Grid)
        K: np.ndarray of floats (nlay, nrow, ncol) or a 3-tuple of such array
            if 3-tuple then the 2 np.ndarrays are kx, ky and kz
                kx  --- array of cell conductivities along x-axis (Ny, Nx, Nz)
                ky  --- same for y direction (if None, then ky=kx )
                kz  --- same for z direction
            else: kx = ky = kz = K
        c: np.ndarray (nlay - 1, nrow, ncol) or None of not used
            Resistance agains vertical flow between the layers [d]
        S: np.ndarray or float
            Storage coefficients for the cells. Currently these are used as a semi-transient simulation
            just to make sure that non-linear boundaries can be smoothly determined.
            It will be straightforward to convert this in a full-scale transient model.
            In the current steady state model, S will be multiplied by gr.Area. and divided
            by dt, where in each outer iteration dt is doubled. This has the same effect
            as gradually enlarging the time in a transient model. It is as if time is doubled
            with every outer iteration, so that the model will approach steady state.
            A 1D sequence will be interpreted as layer values.
        IBOUND: np.ndarray of ints (nlay, nrow, ncol)
            the boundary array like in MODFLOW
            with values denoting:
            * IBOUND > 0  the head in the corresponding cells will be computed
            * IBOUND = 0  cells are inactive, will be given value nan
            * IBOUND < 0  coresponding cells have prescribed head
        HI: np.ndarray (nlay, nrow, ncol)
            Initial heads
            Note that it's IBOUND that determines which HI are fixed heads!
        FQ: np.ndarray of floats (nlay, nrow, ncol)
            Prescribed cell flows (injection positive).
            Every model must have a full size FQ, which may be all zeros.
            
        @TO 20250322
        """
        self.gr = gr
        self.c = c
        self.IBOUND = IBOUND
        self.HI = HI
        self.FQ = FQ
        
        Nz, Ny, Nx = self.gr.shape
        nod = self.gr.nod

        if self.gr.axial is True:
            print("axial==True so that y coordinates and ky are ignored")
            print("            and x stands for r, so that all x coordinates must be >= 0.")
        if isinstance(K, np.ndarray): # only one ndaray was given
            kx, ky, kz = K.copy(), K.copy(), K.copy()
        elif isinstance(K, tuple): # 3-tuple of ndarrays was given
            kx, ky, kz = K[0].copy(), K[1].copy(), K[2].copy()
        else:
            raise ValueError("", "K must be an narray of shape (Ny,Nx,Nz) or a 3tuple of ndarrays")

        if kx.shape != self.gr.shape:
            raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape, self.gr.shape))
        if ky.shape != self.gr.shape:
            raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape, self.gr.shape))
        if kz.shape != self.gr.shape:
            raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape, self.gr.shape))

        self.kx, self.ky, self.kz = kx, ky, kz

        # from this we have the width of columns, rows and layers
        dx = self.gr.dx.reshape(1, 1, Nx)
        dy = self.gr.dy.reshape(1, Ny, 1)
        dz = self.gr.dz.reshape(Nz, 1, 1)

        inact  = (self.IBOUND==0).reshape(nod,) # boolean vector denoting inactive cells
        
        if self.gr.axial is False:
            Rx2 = 0.5 * dx / (dy * dz) / kx
            Rx1 = 0.5 * dx / (dy * dz) / kx
            Ry  = 0.5 * dy / (dz * dx) / ky
            Rz  = 0.5 * dz / (dx * dy) / kz
            if c is not None:
                Rc  =        c / (dx * dy)
            #half cell resistances regular grid
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning) # Division by zero for x=0
                Rx2 = 1 / (2 * np.pi * kx[:,:, 1: ] * dz) * np.log(gr.xm[ 1:]/gr.x[1:-1]).reshape((1, 1, Nx-1))
                Rx1 = 1 / (2 * np.pi * kx[:,:, :-1] * dz) * np.log(gr.x[1:-1]/gr.xm[:-1]).reshape((1, 1, Nx-1))
            Rx2 = np.concatenate((np.inf * np.ones((Nz, Ny, 1)), Rx2), axis=2)
            Rx1 = np.concatenate((Rx1, np.inf * np.ones((Nz, Ny, 1))), axis=2)
            Ry = np.inf * np.ones(self.gr.shape)
            Rz = 0.5 * dz.reshape((Nz, 1, 1))  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)) * kz)
            if c is not None:
                Rc = c  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)))
            #half cell resistances with grid interpreted as axially symmetric

        # set flow resistance in inactive cells to infinite
        Rx2 = Rx2.reshape(nod,)
        Rx2[inact] = np.inf
        Rx2=Rx2.reshape(self.gr.shape)
        
        Rx1 = Rx1.reshape(nod,)
        Rx1[inact] = np.inf
        Rx1=Rx1.reshape(self.gr.shape)
        
        Ry  = Ry.reshape(nod,)
        Ry[ inact] = np.inf
        Ry=Ry.reshape(self.gr.shape)
        
        Rz  = Rz.reshape(nod,)
        Rz[ inact] = np.inf
        Rz=Rz.reshape(self.gr.shape)
        
        #Grid resistances between nodes
        Cx = 1 / (Rx1[:, :,:-1] + Rx2[:, :,1:])
        Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])
        if self.c is None:
            Cz = 1 / (Rz[:-1, :, :] + Rz[:-1, :,:])
        else:
            Cz = 1 / (Rz[:-1, :, :] + Rc + Rz[:-1, :,:])
            
        #Gobal indices for neighboring cells
        IE = self.gr.NOD[:, :, 1: ]  # east neighbor cell numbers
        IW = self.gr.NOD[:, :, :-1] # west neighbor cell numbers
        IN = self.gr.NOD[:, :-1, :] # north neighbor cell numbers
        IS = self.gr.NOD[:,  1:, :]  # south neighbor cell numbers
        IT = self.gr.NOD[:-1, :, :] # top neighbor cell numbers
        IB = self.gr.NOD[ 1:, :, :]  # bottom neighbor cell numbers
        
        def R(x):
            """Shorthand for x.ravel()."""
            return x.ravel()

        # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tuple
        # also notice that Cij = negative but that Cii will be positive, namely -sum(Cij)        
        self.A = sp.csc_matrix((
                -np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\
                (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
                np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\
                        )),(nod, nod))
        
        # Diagonal as a 1D vector
        self.adiag = np.array(-self.A.sum(axis=1))[:,0]
        self.Cx, self.Cy, self.Cz = Cx, Cy, Cz
        
        # Get storage from input line if None or not None
        if S is None:
            Sy, Ss = 0.2, 2e-5
            if gr.nz > 2:
                self.Sto = gr.Volume * Ss
                self.Sto[0] = gr.Area * Sy                
            else:
                self.Sto = gr.AREA * Sy
        elif np.all(gr.shape == S.shape):
            self.Sto = gr.Volume * S
        elif np.isscalar(S):
            self.Sto = gr.AREA * S
        else:       
            S = np.atleast_1d(S).flatten() 
            if len(S) <= gr.nz:
                S = np.pad(S, (0, gr.nz - len(S)), mode="edge")[:, None, None]
                self.Sto    = gr.Volume * S
                self.Sto[0] = gr.Area * S[0]
            else:
                raise ValueError(f"S must have shape {gr.shape} or len nz={gr.nz}")
        return
    
    @staticmethod
    def extend_dtype(data, fields=None):
        """Add fields to the array by extending its dtype.
        
        Parameters
        ----------
        data: np.ndarray with a given dtype
            The original array with its dtype (a.k.a a recarray or a structured array)
        fields: a list of extra fields to be added to the dtype
            e.g. [('this', float), ('that', int64)]        
        """
        new_dtype =data.dtype.descr + fields
        extended_data = np.zeros(data.shape, dtype=new_dtype)
        for name in data.dtype.names:
            extended_data[name] = data[name]
            if name == 'mask': # Fill in True by default
                extended_data[name] == True
        return extended_data
    
    @staticmethod
    def itimize(x):
        """Return scalar if len(x) == 1."""
        if len(x.ravel()) == 1:
            x = x.item()
        return x
    
    @staticmethod
    def mk_fmask(x, transition_width=0.05):
        """Return factor to soften transition when x goes through 0.
        
        The factor varies between 0 and 1 and is > 0.5 voor x > 0
        """
        return 1 / (1 + np.exp(-np.clip(x / transition_width, -50, 50)))
    
    @staticmethod                
    def get_q(dphi, c):
        """Return the net inflow [L/T].
        
        Parameters
        ----------
        dphi: float or np.ndarray
            phi - h (head minus ditch water level).
        c: float
            Drainage resistance (>=0).        
        """
        dphi = np.atleast_1d(dphi).astype(float)
        delta = 0.01
        lam = 2 * delta
        mask = dphi <= delta
        dphi[mask] = delta * np.exp(np.clip(dphi[mask] / lam, -50, 50))
        q = dphi / c
        return Fdm3.itimize(q)
        

    @staticmethod
    def omega(y=None, beta=None):
        """Return wet circumference of parabolic ditch profile given water depth y.
        
        beta is the half-width of the ditch when y = 1
        
        Parameters
        ----------
        y: float or np.ndarray
            Actual ditch water depth.
        beta: float or np.ndarray
            Ditch half width at y=1.
        """
        y = np.atleast_1d(y).astype(float)
        L = np.zeros_like(y) # Half length of Omega

        # In cases where y < yhat (a small value) make sure Omega does not get zero
        delta = 0.01
        yhat = np.e * (delta / beta) ** 2
        mask = y > yhat

        # y > yhat
        b = beta ** 2 / 4
        L[mask] = np.sqrt(y[mask] * (b + y)[mask]) + b[mask] * np.arctanh(np.sqrt(y[mask] / (b + y)[mask]))
        
        # y <= yhat
        lam = 2 * yhat
        L[~mask] = delta * np.exp(np.clip(y[~mask] / lam, -50, 50))
                
        return 2 * Fdm3.itimize(L)

    @ staticmethod
    def cdrainage(y=None, kx=None, kz=None, cy0=None, y0=None, beta=None, L=None):
        """Return the drainage resistance for given ditch water depth.
        
        Parameters
        ----------
        y: float or np.ndarray of float
            Actual ditch water depths.
        kx, ky: floats
            Horizontal and vertical hydraulic conductivities.
        cy0: float or np.ndarray
            Drainage resistance at average recharge.
        y0: float or np.ndarray
            Ditch depth at average recharge.
        beta: float or np.ndarray
            Half width of ditch at y=y0.
        L: float or np.ndarray
            Average distance between ditches.
        
        Returns
        -------
        c: float or np.ndarray
            Actual drainage resistance.        
        """
        y = np.atleast_1d(y).astype(float)
        
        delta_crad = L / (np.pi * np.sqrt(kx * kz)) * np.log(
                    Fdm3.omega(y=y0, beta=beta) / Fdm3.omega(y=y, beta=beta))
        
        cdr = cy0 + delta_crad

        # Prevent negative cdr        
        delta = 0.1 # d (minimum drainage resistance)
        lam = np.e * delta
        cdr[cdr < lam] = delta * np.exp(np.clip(cdr[cdr < lam] / lam, -50, 50))        
        
        return Fdm3.itimize(cdr)
        

        
    def simulate(self, DRN=None, RIV=None, GHB=None, FDR=None, tm=None, htol=1e-7, maxiter=50, verbose=False):
        """Compute a 3D steady state finite diff. model

        Parameters
        ----------
        DRN: array of dtype:
            dtype = dtype([('Ig', 'int'), ('h', float), ('C', float)]) 
        GHB: array ncell of the following dtype:
            dtype = dtype([('Ig', 'int'), ('h', float), ('C', float)])
            Ig = global cell index, h = fixed head and C = conductance
        RIV: array of dtype
            dtype = dtype([('Ig', 'I'), ('h', float), ('C', float)], ('rbot', float)])
            Ig = global index, h = river stage, C = conductance, rbot is river bottom elevation.
        FDR: array of dtype
            dtype([('Ig', 'int'), ('phi, float), ('h', float), ('h0', float), ('N', float), ('w', float), ('L', float)])
            Ig is global index,
            phi = head at mean recharge N
            h = drainage level at mean recharge N
            h0 = bottom elevation of (deepest) ditches
            N = mean recharge
            w = ditch width at mean recharge N
            L = aveage ditch distance.
        tm: None | float | two floats default = [1., 1.5]
            First is the initial time step. The second the time step multiplyer used in outer iterations
            which are needed when non-linear boundary conditions are applied. i.c. DRN, RIV and FDR.
            These time steps are just to stabilize the convergence with non-linear boundary conditions.
            The trick is to simulate transiently with ever increasing time steps to reach steady state.                      
        htol : float (defaults: 1e-5)
            max head difference in successive outer iterations.
        maxiter: int
            Maximum number of outer iterations
            Outer iterations are only applied when non-linear boundaries are used.
        verbose: bool
            If True a plot is made of the convergence progress.

        Returns
        -------
        out: dict
            a dict with fields Phi, cell flow Q, Qx, Qy, Qz, RIV, DRN, GHB, FDR, Qfq, Qfh
            Output shapes are (Nz, Ny, Nx) for Phi and Q and
            (Nz, Ny,Nx-1) for Qx, (Nz,Ny-1,Nx) for Qy, (Nz-1,Ny,Nx) for Qz, while
            DRN, RIV, GHB, FDR are recarrays having a column 'Q' that holds the discharge for the the given nodes.
            Qfq is a full array of gr.shape and Qfh is a recarray with fields 'Ig' and 'Q'

        TO 160905,, 240316, 250320
        """
        nod = int(self.gr.nod)

        #cell numbers for neighbors
        active = (self.IBOUND > 0).reshape(nod,)  # boolean vector denoting the active cells
        inact  = (self.IBOUND ==0).reshape(nod,) # boolean vector denoting inacive cells
        fxhd   = (self.IBOUND < 0).reshape(nod,)  # boolean vector denoting fixed-head cells


        # DRN boundaries
        if DRN is not None:
            if not DRN.dtype == self.__class__.dtype['drn']:
                raise ValueError(
                    f"""DRN must have dtype:\n{self.__class__.dtype['drn']}\nnot\n{DRN.dtype}"""
                )
            DRN = self.__class__.extend_dtype(DRN, fields=[('Q', float), ('phi', float), ('fmask', float)])
            
        # RIV boundaries
        if RIV is not None:
            if not RIV.dtype == self.__class__.dtype['riv']:
                raise ValueError(
                    f"""RIV must have dtype:\n{self.__class__.dtype['riv']}\nnot\n{RIV.dtype}"""
                )
            RIV = self.__class__.extend_dtype(RIV, fields=[('Q', float), ('phi', float), ('fmask', float)])

            
        # General head boundaries
        if GHB is not None:
            if not GHB.dtype == self.__class__.dtype['ghb']:
                raise ValueError(
                    f"""GHB must have dtype:\n{self.__class__.dtype['ghb']}\nnot\n{GHB.dtype}""")
            GHB = self.__class__.extend_dtype(GHB, fields=[('Q', float), ('phi', float),('fmask', float)])
            
        # Free drainage boundaries (kind of drains)
        if FDR is not None:
            if not FDR.dtype == self.__class__.dtype['fdr']:
                raise ValueError(
                    f"""FDR must have dtype:\n{self.__class__.dtype['fdr']}\nnot\n{FDR.dtype}"""
                )
            FDR = self.__class__.extend_dtype(FDR, fields=[('Q', float), ('C', float), ('c', float),
                                                             ('gamma', float), ('eta', float), ('beta', float), ('ge', float), ('fmask', float)])    
                
            # Compute the initial coefficients taking phi=phiN, h=hN, q=N
            # Make sure phiN > hN and hN > h0, N>0, so that dphi>0 and y>0 initially
            

        non_linear_options = [name for (name, p) in zip(['DRN', 'RIV', 'FDR'], [DRN, RIV, FDR]) if p is not None]
        
        # Initial heads
        HI = self.HI.copy().reshape(nod, 1)
        
        # Set up the initial head vector
        Phi = HI[:, 0].copy()

        # Outer iterations get first dt and step multiplier
        if tm is None:
            dt, dtmult = 1., 2.
        elif np.isscalar(tm):
            dt, dtmult = tm, 2.
        else:            
            dt, dtmult = tm[0], tm[1] / tm[0]        
        
        
        if verbose:
            # Make a plot of the progress of phi
            _, ax = plt.subplots(figsize=(10, 6))
            title=f"Phi during the outer iterations: non linear options ={non_linear_options}"
            ax.set(title=title, xlabel='x', ylabel='Phi', xscale='linear', yscale='linear')        
                
        for iouter in range(maxiter):

            # Keep A's diagonal as a 1D vector.
            # Restart fresh in each loop cycle
            adiag = self.adiag.copy()

            # Set up up the Right-hand side vector, using FQ.
            # Restart fresh in each loop cycle
            RHS = self.FQ.copy().reshape(nod, 1)

            if np.any(fxhd):
                RHS -= (self.A + sp.diags(adiag, 0))[:, fxhd] @ HI[fxhd]

            # Add the boundary conditions
            if DRN is not None:
                # Drains as in Modflow
                Ig = DRN['Ig']
                DRN['phi'] = Phi[Ig]                
                DRN['fmask'] = self.__class__.mk_fmask(DRN['phi'] - DRN['h'])
                adiag[Ig]  += DRN['fmask'] * DRN['C']
                RHS[Ig, 0] += DRN['fmask'] * DRN['C'] * DRN['h']

            if RIV is not None:
                # River cells as in Modflow
                Ig =RIV['Ig']
                RIV['phi'] = Phi[Ig]                   
                if True: # Non-linear
                    RIV['fmask'] = self.__class__.mk_fmask(Phi[Ig] - RIV['rbot'])                    
                    adiag[Ig]  += RIV['C'] * RIV['fmask']
                    RHS[Ig, 0] += RIV['C'] * (RIV['fmask'] * RIV['h'] + (1 - RIV['fmask']) * (RIV['h'] - RIV['rbot']))
                else: # L inear
                    adiag[Ig] += RIV['C']
                    RHS[Ig, 0] += RIV['C'] * RIV['h']

            if GHB is not None:
                # GHB cells as in Modflow.
                # No outer iterations are needed.
                Ig = GHB['Ig']
                GHB['phi'] = Phi[Ig]
                adiag[Ig] += GHB['C']
                RHS[  Ig, 0] += GHB['C'] * GHB['h']

            if FDR is not None:
                # Free drainage situation.
                # As mathematically derived.               
                Ig = FDR['Ig']
                math_cdr = np.any(FDR['w'] == 0.) or np.any(np.isnan(FDR['w'])) or np.any(FDR['L'] == 0.) or np.any(np.isnan(FDR['L']))               
                if iouter == 0:
                    if math_cdr:
                        print("Using FDR, with mathematical drainage.")
                    else:
                        print("Using FDR with physical drainage.")
                        
                    # Initialize, using the initial values passed in with FDR                    
                    
                    # Validate input
                    if not np.all((FDR['phi'] > FDR['h']) & (FDR['h'] > FDR['h0'])):
                        raise ValueError("Initially we must have FDR['phi'] > FDR['h] > FDR['h0']")
                    if not np.all(FDR['N'] > 0):
                        raise ValueError("Initially we must have FDR['N'] > 0")
                    
                    y0 = FDR['h'] - FDR['h0'] # Initial water depth in ditch.
                    FDR['eta'  ] = y0 / np.sqrt(FDR['N'])    # eta in y = eta sqrt(q)                
                    FDR['c'] = (FDR['phi'] - FDR['h']) / FDR['N'] # Initial drainage resistance
                    cy0  = FDR['c'].copy() # We need y0 also in subsequent cycles
                    if math_cdr:
                        # Using the mathematical drainage resistance approach
                        FDR['gamma'] = FDR['c'] * np.sqrt(FDR['N'])                        
                        FDR['ge'] = (FDR['gamma'] + FDR['eta']) ** 2                        
                        FDR['C'] = self.gr.Area.ravel()[Ig] * (FDR['phi'] - FDR['h0']) / FDR['ge']
                    
                    else:
                        # Using the physical drainage resistance approach
                        FDR['beta']  = (FDR['w'] / 2) / np.sqrt(y0)
                        kx = self.kx.ravel()[Ig]
                        kz = self.kz.ravel()[Ig]                        
                
                else:
                    # During further cycles of the outer loop
                    FDR['phi'] = Phi[Ig]
                    FDR['fmask'] = Fdm3.mk_fmask(Phi[Ig] - FDR['h0'])                    
                    
                    if math_cdr:  
                        FDR['C'] = self.gr.Area.ravel()[Ig] * FDR['fmask'] / FDR['ge']
                    
                    else: # phisical drainage formula
                        q = Fdm3.get_q(FDR['phi'] - FDR['h'], FDR['c'])
                        y = FDR['eta'] * np.sqrt(q)
                        FDR['c'] = Fdm3.cdrainage(y=y, kx=kx, kz=kz, cy0=cy0, y0=y0, beta=FDR['beta'], L=FDR['L'])
                        FDR['C'] = self.gr.Area.ravel()[Ig] / (FDR['c'] + FDR['eta'] / np.sqrt(q))
                FDR['fmask'] = Fdm3.mk_fmask(FDR['phi'] - FDR['h0'])
                adiag[Ig] +=  FDR['C']
                RHS[Ig, 0] += FDR['C'] * FDR['h0']
                
            # Quasi time stepping to stabilize non linear behavior

            if non_linear_options: # We loop using outer iterations
                if iouter == 0:
                    print(f"Non-linear options: {non_linear_options}, starting outer iterations:")
                # Add storage to stabilize the solution (make it transient to reach steady state)
                adiag += self.Sto.ravel() / dt
                RHS[:, 0] += self.Sto.ravel() / dt * HI[:, 0]                
                
                Phi[active] = la.spsolve((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active] )                
                # Phi[active], info = la.cg((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active], x0=HI[active], rtol=rtol, atol=atol, maxiter=250)                
                err = np.abs(Phi[active] - HI.ravel()[active]).max()
                errBalance = (self.Sto.ravel() * (Phi - HI.ravel())).sum()
                
                print(f"iouter = {iouter:4}, err = {err:10.5g} m, errBalance = {errBalance:10.5g}")                
                if err < htol:
                    print("Converged, normal termination.")
                    break
                elif iouter == maxiter - 1:                    
                    print("Not converged!")
                else:
                    if verbose:
                        ax.plot(self.gr.xm, Phi, label=f'iouter={iouter}')
                    HI[:, 0] = Phi
                    dt *= dtmult                     
            else:
                Phi[active] = la.spsolve((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active] )                
                print("No outer iterations needed.")
                break

        if verbose:
            ax.grid()
            ax.legend()
        
        # Prepare output, all items are returned in a single dictionary out
        out=dict()
        out.update(gr=self.gr)
        
        # reshape Phi to shape of grid
        Phi = Phi.reshape(self.gr.shape)
        out.update(Phi=Phi)

        # Net cell inflow
        Q = (np.array((self.A + sp.diags(self.adiag, 0)) @ Phi.ravel())).reshape(self.gr.shape)
        out.update(Q=Q)
        
        #Flows across cell faces
        Qx =  -np.diff(Phi, axis=2) * self.Cx
        Qy =  +np.diff(Phi, axis=1) * self.Cy
        Qz =  +np.diff(Phi, axis=0) * self.Cz
        out.update(Qx=Qx, Qy=Qy, Qz=Qz)
                
        # Put Qfh in an array met dtype to keep their Ig numbers.
        nfh = fxhd.sum()        
        Qfh = np.zeros(nfh, dtype=np.dtype([('Ig', int), ('Q', float)]))
        Qfh['Ig'] = self.gr.NOD.ravel()[fxhd]
        Qfh['Q']  = Q.ravel()[Qfh['Ig']]
        
        out.update(Qfq=self.FQ, Qfh=Qfh)
        
        if DRN is not None:                                        
            DRN['Q'] = DRN['fmask'] * DRN['C'] * (DRN['h'] - Phi.ravel()[DRN['Ig']])
            out.update(DRN=DRN)

        if RIV is not None:            
            RIV['Q'] = ( RIV['fmask']  * RIV['C'] * (RIV['h'] - Phi.ravel()[RIV['Ig']]) + 
                    (1 - RIV['fmask']) * RIV['C'] * (RIV['h'] - RIV['rbot'])
            )
            out.update(RIV=RIV)

        if GHB is not None:            
            GHB['Q'] = GHB['C'] * (GHB['h'] - Phi.ravel()[GHB['Ig']])
            out.update(GHB=GHB)

        if FDR is not None:                        
            FDR['Q'] = FDR['fmask'] * FDR['C'] * (FDR['h0'] - Phi.ravel()[FDR['Ig']])
            out.update(FDR=FDR)

        # Finally: set inactive cells to np.nan
        out['Phi'][inact.reshape(self.gr.shape)] = np.nan # put np.nan at inactive locations
        
        watbal(out)
        return out
    
def watbal(out):
    """Print the water budget of the model divided according to boundary types.
    
    The resulting Q1 is the net inflow in each cell based on A @ RHS, internal flows from each cell to each neighbors.
    Q2 is the total flow into the model by all boundary types.
    Q2 is then split over Qfh, Qfq, Qghb, Qriv, Qdrn and Qdrn1.
    
    All values are model totals.
    Q1 should be close to Q2

    @TO 20250320

    Parameters
    ----------
    out: dictionary
        Output of Fem3D simulation.
    """
    gr = out['gr']
    
    print()
    print("===== Water balance of the entire model =====")
    print(f"Model grid = {gr.shape}")
    
    Q1 = out['Q'].sum()
    print()
    print("Sum over all nodal flows should be zero:")
    print(f"Total net in Q1     = {Q1:10.5g} m3/d")
    
    Q2 = 0.
    Qfq = out['Qfq'].sum()
    Q2 += Qfq
    
    Qfh = out['Qfh']['Q'].sum()
    Q2 += Qfh
    
    print()
    print("Boundary components, these components should also add up zo zero:")
    print(f"Fixed flows         FQ   = {Qfq:15.3f} m3/d")
    print(f"Fixed heads         QFH  = {Qfh:15.3f} m3/d")

    if 'GHB' in out:
        Qghb = out['GHB']['Q'].sum()
        print(f"GHB                Qghb = {Qghb:15.3f} m3/d, flow from GHB cells.")
        Q2 += Qghb
    
    if 'DRN' in out:
        Qdrn = out['DRN']['Q'].sum()
        print(f"DRN                Qdrn  = {Qdrn:15.3f} m3/d")
        Q2 += Qdrn
        
    if 'RIV' in out:
        Qriv = out['RIV']['Q'].sum()
        print(f"RIV                Qriv  = {Qriv:15.3f} m3/d, flow from RIV cells.")
        Q2 += Qriv
        
    if 'FDR' in out:
        Qfdr = out['FDR']['Q'].sum()
        print(f"FDR                Qfdr = {Qfdr:15.3f} m3/d, flow from FDR cells.")
        Q2 += Qfdr
    
    print()
    print("Total water balance of this model")
    print(f"Q1 (internal)            = {Q1:15.3f} m3/d, total internal flow")
    print(f"Q2 (boundaries)          = {Q2:15.3f} m3/d, total from boundaries")
    print("===== end of water balance =====\n")
    return

def mazure(kw):
    """1D flow in semi-confined aquifer example
    Mazure was Dutch professor in the 1930s, concerned with leakage from
    polders that were pumped dry. His situation is a cross section perpendicular
    to the dike of a regional aquifer covered by a semi-confining layer with
    a maintained head in it. The head in the regional aquifer at the dike was
    given as well. The head obeys the following analytical expression
    phi(x) - hp = (phi(0)-hp) * exp(-x/B), B = sqrt(kDc)
    To compute we use 2 model layers and define the values such that we obtain
    the Mazure result.
    """
    z = kw['z0'] - np.cumsum(np.hstack((0., kw['D'])))
    gr = mfgrid.Grid(kw['x'], kw['y'], z, axial=False)
    c = gr.const(kw['c'])
    k = gr.const(kw['k'])
    kD = (kw['k'] * kw['D'])[-1] # m2/d, transmissivity of regional aquifer
    B = np.sqrt(kD * float(kw['c'])) # spreading length of semi-confined aquifer
    
    FQ = gr.const(0.) # prescribed flows
    
    s0 = 2.0 # head in aquifer at x=0
    HI = gr.const(0); HI[-1, :, 0] = s0 # prescribed heads
    
    IBOUND = gr.const(1); IBOUND[0, :, :] = -1; IBOUND[-1, :, 0]=-1
    
    mdl = Fdm3(gr=gr, K=k, c=c, IBOUND=IBOUND, FQ=FQ, HI=HI)
    out = mdl.simulate()
    
    ax = newfig(kw['title'], 'x[m]', 's [m]')
    
    ax.plot(gr.xm, out['Phi'][-1, 0 ,:], 'r.', label='fdm3')
    ax.plot(gr.xm, s0 * np.exp(-gr.xm / B),'b-', label='analytic')
    ax.legend()
    return out


def deGlee(kw):
    """Simulate steady axial flow to fully penetrating well in semi-confined aquifer"""
    z = kw['z0'] - np.cumsum(np.hstack((0., kw['D'])))
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))
    gr = mfgrid.Grid(r, None, z, axial=True)
    k = gr.const(kw['k'])
    c = gr.const(kw['c'])
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    lambda_  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    FQ = gr.const(0.)
    FQ[-1, 0, 0] = kw['Q']   # m3/d fixed flows
    HI = gr.const(0.)                           # m, initial heads
    
    IBOUND = gr.const(1); IBOUND[0, :, :] = -1  # modflow like boundary array
    
    mdl = Fdm3(gr=gr, K=k, c=c, IBOUND=IBOUND, FQ=FQ, HI=HI) # run model
    out = mdl.simulate() # run model
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=kw['title'], xlabel='r [m]', ylabel='s [m]', xscale='log', xlim=[1e-3, r[-1]])
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * K0(gr.xm / lambda_) / (kw['rw']/ lambda_ * K1(kw['rw']/ lambda_)), 'b-',label='analytic')
    ax.grid()
    ax.legend()
    return out
    
    
def axial_all_boundary_types(kw):
    """Run Axial symmetric examples, but now using GHB each of the boundary types instead,
    just a single layer so simulate drainage:
        GHB General head boundaries
        DRN drains.
        RIV river cells. Setting h=rbot, so that RIV is equivalent to DRN
        FDR free drainage, once using math resitance, once using physical drainage resistance
    
    Simulating leakage using boundary types instead of a resistance layer. Allows using only a single
    aquifer.
    With a single aquifer, resistance layers (c) cannot be used.
    """
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    lambda_  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    # Set up the grid
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))
    z = (kw['z0'] - np.cumsum(np.hstack((0., kw['D']))))[1:] # no top layer
    gr = mfgrid.Grid(r, None, z, axial=True)   # generate grid
    
    FQ = gr.const(0.)                   # m3/d fixed flows
    FQ[0] = gr.Area * kw['N']           # Recharge areal
    FQ[-1, 0, 0] += kw['Q']             # Add extraction   
    HI = gr.const(0.)                   # m, initial heads
    
    IBOUND = gr.const(1, dtype=int)     # modflow like boundary array
    
    k = gr.const(kw['k'][-1])           # full 3D array of hydraulic conductivities
    
    # Set up the various boundary types to be used in turn for comparison
    Ig = gr.NOD[0].ravel() # Select the cells that will be part of the boundary
    
    GHB = np.zeros(len(Ig), dtype=Fdm3.dtype['ghb'])
    GHB['Ig'], GHB['h'], GHB['C'] = Ig, HI.ravel()[Ig], gr.Area.ravel() / kw['c']
        
    DRN = GHB.copy() # Same parameters
    
    RIV = np.zeros(len(Ig), dtype=Fdm3.dtype['riv'])
    RIV['Ig'], RIV['h'], RIV['C'], RIV['rbot'] = Ig, HI.ravel()[Ig], gr.Area.ravel() / kw['c'], HI.ravel()[Ig]
    
    # Free drainage
    FDR = np.zeros(len(Ig), dtype=Fdm3.dtype['fdr'])
    phi, Nc = HI.ravel()[Ig], kw['N'] * kw['c']
    FDR['Ig'], FDR['phi'], FDR['h'], FDR['h0'], FDR['N'], FDR['w'] = Ig, phi, phi - Nc, phi - Nc - kw['y0'], kw['N'], kw['w1']
    
    mdl = Fdm3(gr=gr, K=k, c=None, IBOUND=IBOUND, HI=HI, FQ=FQ) # run model    
    out_ghb = mdl.simulate(GHB=GHB) # run model
    out_drn = mdl.simulate(DRN=DRN, tm=[1., 5.]) # run model
    out_riv = mdl.simulate(RIV=RIV, tm=[1., 5.]) # run model
    out_fdr1 = mdl.simulate(FDR=FDR, tm=[1., 5.]) # run model
    
    FDR['w'] = kw['w2']
    out_fdr2 = mdl.simulate(FDR=FDR, tm=[1., 5.]) # run model
    
    title = kw['title'] + ' (Using GHB)'
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=title, xlabel='r [m]', ylabel='s [m]', xscale='log', xlim=[1e-3, r[-1]])
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * K0(gr.xm / lambda_) / (kw['rw']/ lambda_ * K1(kw['rw']/ lambda_)), '.',label='De Glee Analytic')
    ax.plot(gr.xm, out_ghb[ 'Phi'][-1, 0, :], '--', lw=1, label='Fdm3 GHB')
    ax.plot(gr.xm, out_drn[ 'Phi'][-1, 0, :], '-.', lw=1, label='Fdm3 DRN')
    ax.plot(gr.xm, out_riv[ 'Phi'][-1, 0, :], ':',  lw=1, label='Fdm3 RIV, h=rbot')
    ax.plot(gr.xm, out_fdr1['Phi'][-1, 0, :], '-',  lw=1, label='Fdm3 FDR, math. drainage')
    ax.plot(gr.xm, out_fdr2['Phi'][-1, 0, :], '-',  lw=1, label='Fdm3 FDR, phys. drainage')
        
    ax.grid()
    ax.legend()
    return {'ghb': out_ghb, 'drn': out_drn, 'riv': out_riv, 'fdr1': out_fdr1, 'fdr2': out_fdr2}

cases = {
    'Mazure': {
        'title': 'Mazure 1D flow',
        'comment': """1D flow in semi-confined aquifer example
        Mazure was Dutch professor in the 1930s, concerned with leakage from
        polders that were pumped dry. His situation is a cross section perpendicular
        to the dike of a regional aquifer covered by a semi-confining layer with
        a maintained head in it. The head in the regional aquifer at the dike was
        given as well. The head obeys the following analytical expression
        phi(x) - hp = (phi(0)-hp) * exp(-x/B), B = sqrt(kDc)
        To compute we use 2 model layers and define the values such that we obtain
        the Mazure result.
        """,
        'z0': 0.,
        'x': np.hstack((0.001, np.linspace(0., 2000., 101))), # column coordinates
        'y': np.array([-0.5, 0.5]), # m, model is 1 m thick
        'D': np.array([10., 50.]), # m, thickness of confining top layer
        'c': np.array([[250.]]), # d, vertical resistance of semi-confining layer
        'k': np.array([np.inf, 10.]),        
        },
    'axial_data': {
        'title': 'Axially symmetric flow',
        'comment': """Axial symmetric examples. All for a well simulating
                extraction in a aquifer with drainage. The drainage is simulated
                using different boundary conditions, both linear and non-linear:
                GHB: Gneral Head Boundaries.            
                DRN: drain boundaries. (Non linear).
                RIV: riv boundaries. (Non linear, made idential to DRN by setting h=rbot).
                FDR: free drainage boundaries
                    Version one: drainage resistance using a math root function.
                    Version two: drainage resistance using a physical function with ditch width.
                """,
        'z0': 0.,
        'Q': -2500.,
        'rw':   .25,
        'D' : np.array([10., 50.]),
        'c' :  np.array([[1000.]]),
        'k' :  np.array([np.inf, 10]),  # m/d conductivity of regional aquifer
        'r' : np.logspace(-2, 4, 61),  # distance to well center        
        'N' : 0.001, # Reference recharge
        'y0': 1.0,   # Ditch depth for reference situation        
        'w1' : np.nan, # Use math function for drainage resistance
        'w2' : 1.0,     # Ditch width, use physical formula for dr. res.
    },
}  

if __name__ == "__main__":
    # out1 = mazure(cases['Mazure'])
    out_all = axial_all_boundary_types(cases['axial_data'])
    
    for name, out in out_all.items():
        print(name)
        watbal(out)
        
    plt.show()
    
