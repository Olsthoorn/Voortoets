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
import scipy.special as spec
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


def fdm3(gr=None, K=None, c=None, S=None, t=None, FQ=None, HI=None, IBOUND=None, DRN=None, RIV=None, GHB=None, DRN1=None, tolh=0.0001, maxiter=50):
    '''Compute a 3D steady state finite diff. model

    Parameters
    ----------
    gr: mfgrid.Grid object instance
        object holding the grid/mesh (see mfgrid.Grid)
    K: np.ndarray of floats (nlay, nrow, nocl) or a 3-tuple of such array
        if 3-tuple then the 2 np.ndarrays are kx, ky and kz
            kx  --- array of cell conductivities along x-axis (Ny, Nx, Nz)
            ky  --- same for y direction (if None, then ky=kx )
            kz  --- same for z direction
        else: kx = ky = kz = K
    c: np.ndarray (nlay - 1, nrow, ncol) or None of not used
        Resistance agains vertical flow between the layers [d]
    S: Storage coefficient per cell, will be multiplied by cell area.
        Only used here to stabilize solutions with non-linear boundaries
        Default Sy=0.2 in top layer, S=0.001 in lower layers.
    t: np.ndarray of increasing times or float
        times from which time steps are obtained.
        t[0] or the float will be the first time step.
        t is only used here to stabilize non-linear boundaries.
        Default = 1 d.
    DRN: array of dtype:
        dtype = dtype([('Ig', 'O'), ('h', float), ('C', float)]) 
    GHB: array (ncell, 3)
        dtype = dtype([('Ig', 'O'), ('hbound', float), ('C', float)])
        list of sequence, cellid may also be the global cell index
        [[(k,j ,i), h, cond],
         [...]]
    RIV: array of 
        dtype = dtype([('Ig', 'O'), ('h', float), ('C', float)], ('rbot', float)])         
    DRN1: array of
        dtype([('Ig', 'O'), ('phi, float), ('hN', float), ('h0', float), ('N', float), ('wN', float)])
         phi = head at mean recharge N
         hN = drainage level at mean recharge N
         h0 = bottom elevation of (deepest) ditches
         N = average recharge
         wN = ditch width at mean recharge N    
    FQ: np.ndarray of floats (nlay, nrow, ncol)
        Prescrived cell flows (injection positive)
    IH: np.ndarray (nlay, nrow, ncol)
        Initial heads
    IBOUND: np.ndarray of ints (nlay, nrow, ncol)
        the boundary array like in MODFLOW
        with values denoting:
        * IBOUND>0  the head in the corresponding cells will be computed
        * IBOUND=0  cells are inactive, will be given value nan
        * IBOUND<0  coresponding cells have prescribed head
        
    Returns
    -------
    out: dict
        a dict with fields Phi, Qx, Qy, Qz and cell flow Q
        Output shapes are
        (Ny,Nx,Nz) (Ny,Nx-1,Nz), (Ny-1,Nx,Nz), (Ny,Nx,Nz-1), (Ny, Nx, Nz)

    TO 160905,, 240316
    '''
    
    # dtypes for the boundary types (alike those of flopy)    
    drn_dtype = np.dtype([('Ig', 'O'), ('h', float), ('C', float)])
    riv_dtype = np.dtype([('Ig', 'O'), ('h', float), ('C', float), ('rbot', float)]
                         )
    ghb_dtype = np.dtype([('Ig', 'O'), ('bhead', float), ('C', float)])
    
    # For free runoff
    drn1_dtype = np.dtype([('Ig', 'O'), ('phi', float), ('hN', float),
                           ('h0', float), ('N', float), ('wN', float)]
                          )
    # ---------------------------------

    Nz, Ny, Nx = SHP = gr.shape
    nod = Ny * Nx * Nz
    NOD = np.arange(nod).reshape(SHP) # generate cell numbers

    if gr.axial is True:
        print("axial==True so that y coordinates and ky are ignored")
        print("            and x stands for r, so that all x coordinates must be >= 0.")
    if isinstance(K, np.ndarray): # only one ndaray was given
        kx, ky, kz = K.copy(), K.copy(), K.copy()
    elif isinstance(K, tuple): # 3-tuple of ndarrays was given
        kx, ky, kz = K[0].copy(), K[1].copy(), K[2].copy()
    else:
        raise ValueError("", "K must be an narray of shape (Ny,Nx,Nz) or a 3tuple of ndarrays")

    if kx.shape != SHP:
        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape, SHP))
    if ky.shape != SHP:
        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape, SHP))
    if kz.shape != SHP:
        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape, SHP))

    # from this we have the width of columns, rows and layers
    dx = gr.dx.reshape(1, 1, Nx)
    dy = gr.dy.reshape(1, Ny, 1)
    dz = gr.dz.reshape(Nz, 1, 1)

    active = (IBOUND>0 ).reshape(nod,)  # boolean vector denoting the active cells
    inact  = (IBOUND==0).reshape(nod,) # boolean vector denoting inacive cells
    fxhd   = (IBOUND<0 ).reshape(nod,)  # boolean vector denoting fixed-head cells

    if gr.axial is False:
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
        Ry = np.inf * np.ones(SHP)
        Rz = 0.5 * dz.reshape((Nz, 1, 1))  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)) * kz)
        if c is not None:
            Rc = c  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)))
        #half cell resistances with grid interpreted as axially symmetric

    # set flow resistance in inactive cells to infinite
    Rx2 = Rx2.reshape(nod,); Rx2[inact] = np.inf; Rx2=Rx2.reshape(SHP)
    Rx1 = Rx1.reshape(nod,); Rx1[inact] = np.inf; Rx1=Rx1.reshape(SHP)
    Ry  = Ry.reshape( nod,); Ry[ inact] = np.inf; Ry=Ry.reshape(SHP)
    Rz  = Rz.reshape( nod,); Rz[ inact] = np.inf; Rz=Rz.reshape(SHP)
    #Grid resistances between nodes

    Cx = 1 / (Rx1[:, :,:-1] + Rx2[:, :,1:])
    Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])
    if c is None:
        Cz = 1 / (Rz[:-1, :, :] + Rz[:-1, :,:])
    else:
        Cz = 1 / (Rz[:-1, :, :] + Rc + Rz[:-1, :,:])
        
    
    # DRN boundaries
    if DRN is not None:
        if not DRN.dtype == drn_dtype:
            raise ValueError(
                f"""DRN must have dtype:\n{drn_dtype}\nnot\n{DRN.dtype}"""
            )        
        
    # RIV boundaries
    if RIV is not None:
        if not RIV.dtype == riv_dtype:
            raise ValueError(
                f"""RIV must have dtype:\n{riv_dtype}\nnot\n{RIV.dtype}"""
            )

        
    # General head boundaries
    if GHB is not None:
        if not GHB.dtype == ghb_dtype:
            raise ValueError(
                f"""GHB must have dtype:\n{ghb_dtype}\nnot\n{GHB.dtype}""")
        
    # Free drainage GHB boundaries (kind of drains)
    if DRN1 is not None:
        if not DRN1.dtype == drn1_dtype:
            raise ValueError(
                f"""DRN1 must have dtype:\n{drn1_dtype}\nnot\n{DRN1.dtype}"""
            )       
            
        # Compute the coefficients
        DRN1['gamma'] = (DRN1['phi'] - DRN1['hN']) / np.sqrt(DRN1['N'])
        DRN1['eta']   = (DRN1['hN'] - DRN1['n0']) / np.sqrt(DRN1['N'])
        DRN1['beta']  = (DRN1['wd'] / 2) / np.sqrt(DRN1['hN'] - DRN1['h0'])
        
    #conductances between adjacent cells

    IE = NOD[:, :, 1: ]  # east neighbor cell numbers
    IW = NOD[:, :, :-1] # west neighbor cell numbers
    IN = NOD[:, :-1, :] # north neighbor cell numbers
    IS = NOD[:,  1:, :]  # south neighbor cell numbers
    IT = NOD[:-1, :, :] # top neighbor cell numbers
    IB = NOD[ 1:, :, :]  # bottom neighbor cell numbers
    #cell numbers for neighbors

    def R(x):
        """Shorthand for x.ravel()."""
        return x.ravel()

    # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tuple
    # also notice that Cij = negative but that Cii will be positive, namely -sum(Cij)
    A = sp.csc_matrix((
            -np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\
            (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
             np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\
                      )),(nod,nod))


    for iter in range(maxiter):
        # Keep A's diagonal as a 1D vector
        adiag = np.array(-A.sum(axis=1))[:,0]

        # St up up the Right-hand side vector, just using FQ, which is a gr.shape array.
        RHS = FQ.reshape(nod,1) - (
            A + sp.diags(adiag, 0))[:,fxhd] @ HI[fxhd]

        # Set up the initial head vector    
        Phi = HI.flatten()
        
        # Add the boundary conditions
        if DRN is not None:
            Ig = DRN['Ig']
            mask = Phi[Ig] > DRN['h'] # select cells that actually leak             
            adiag[Ig][mask] += DRN['h'][mask]
            RHS[  Ig][mask] += DRN['C'][mask] * DRN['h'][mask]
            
        if RIV is not None:
            Ig =RIV['Ig']
            mask = Phi[Ig] > RIV['rbot']        
            adiag[Ig][ mask] += RIV['C'][mask]
            RHS[  Ig][ mask] += RIV['C'][mask] * RIV['h'][mask]
            RHS[  Ig][~mask] += RIV['C'] * (RIV['h'][~mask] - RIV['rbot'][~mask])        

        if GHB is not None:
            Ig = GHB['Ig']
            adiag[Ig] += GHB['C']
            RHS[  Ig] += GHB['C'] * GHB['hbound']

        if DRN1 is not None:
            delta = 0.001 # m
            Ig = DRN1['Ig']
            mask = Phi[Ig] > DRN1['h'] # Initially h = hN
            DRN1['q'][mask] = (Phi[Ig][mask] - DRN1['h'][mask]) / DRN1['c'][mask]
            DRN1['h'] = delta + DRN1['h0'] + DRN1['eta'][mask] * np.sqrt(DRN1['q'])
            DRN1['C'][mask] = gr.Area[Ig][mask] * np.sqrt(DRN1['q'][mask]) / DRN1['gamma'][mask]

            adiag[Ig][mask] += DRN1['C'][mask]
            RHS[Ig][mask] += DRN1['C'][mask] * DRN1['h'][mask]
        
        Phi[active] = la.spsolve((A + la.spdiags(adiag, 0, gr.nod, gr.nod))[active][:,active] ,RHS[active] )
        
        err = np.abs(Phi[active] - h[active]).max()
        print(iter, err)
        if err < tolh:            
            break
        
    print(f"finally iter={iter}, max head change ={err:.4g}")
        

    # reshape Phi to shape of grid
    Phi = Phi.reshape(gr.shape)

    # Net cell inflow
    Q = gr.const(0.)
    Q[active]  = A[active][:,active] @ Phi[active]
    Q = Q.reshape(gr.shape)

    #Flows across cell faces
    Qx =  -np.diff(Phi, axis=2) * Cx
    Qy =  +np.diff(Phi, axis=1) * Cy
    Qz =  +np.diff(Phi, axis=0) * Cz
    
    out=dict()
    out.update(Phi=Phi, Q=Q, Qx=Qx, Qy=Qy, Qz=Qz)
    
    if DRN is not None:
        DRN['Q'] = 0
        mask = Phi[DRN['Ig']] > DRN['h']
        DRN['Q'][mask] = DRN['C'][mask] * (DRN['h'][mask] - Phi[DRN['Ig'][mask]])
        Qdrn = gr.const(0.)
        Qdrn[DRN['Ig']] = DRN['Q']     
        out.update(Qdrn=Qdrn)
    if RIV is not None:
        RIV['Q'] = 0.
        mask = Phi[RIV['Ig']] > RIV['rbot']
        RIV['Q'][ mask] = RIV['C'][ mask] * (RIV['h'][ mask] - Phi[RIV['cellID'][mask]])
        RIV['Q'][~mask] = RIV['C'][~mask] * (RIV['h'][~mask] - RIV['rbot'][~mask])
        Qriv = gr.cont(0.)
        Qriv.ravel()[RIV['Ig']] = RIV ['Q']     
        out.update(Qriv=Qriv)
        pass
    if GHB is not None:
        GHB['Q'] = 0.
        GHB['Q'] = GHB['C'] * (GHB['h'] - Phi[DRN['Ig']])
        Qghb = gr.const(0.)
        Qghb[GHB['Ig']] = GHB['Q']
        out.update(Qghb=Qghb)
    if DRN1 is not None:
        DRN1['Q'] = 0.
        mask = Phi[DRN1['Ig']] > DRN1['h0']
        DRN1['Q'][mask] = DRN1['C'][mask] * (DRN1['h'][mask] - Phi[DRN1['Ig'][mask]])
        Qdrn1 = gr.const(0.)
        Qdrn1.ravel()[DRN1['Ig']] = DRN1['Q']
        out.update(Qdrn1=Qdrn1.ravel)
        pass

        # set inactive cells to nan
    out['Phi'][inact.reshape(gr.shape)] = np.nan # put nan at inactive locations

    return out

# --------
class Fdm3():
    """Finite difference model class."""

    def __init__(self, gr=None, K=None, c=None, S=None, IBOUND=None, HI=None, FQ=None):
        """Return intantiated model with grid and aquifer properties but without its boundaries.
        
        Parameters
        ----------
        gr: mfgrid.Grid object instance
            object holding the grid/mesh (see mfgrid.Grid)
        K: np.ndarray of floats (nlay, nrow, nocl) or a 3-tuple of such array
            if 3-tuple then the 2 np.ndarrays are kx, ky and kz
                kx  --- array of cell conductivities along x-axis (Ny, Nx, Nz)
                ky  --- same for y direction (if None, then ky=kx )
                kz  --- same for z direction
            else: kx = ky = kz = K
        c: np.ndarray (nlay - 1, nrow, ncol) or None of not used
            Resistance agains vertical flow between the layers [d]
        S: np.ndarray or float
            Storage coefficients for the cells.
            Will be multiplied by gr.Area.
            A 1D sequence will be interpreted as layer values.
        IBOUND: np.ndarray of ints (nlay, nrow, ncol)
            the boundary array like in MODFLOW
            with values denoting:
            * IBOUND>0  the head in the corresponding cells will be computed
            * IBOUND=0  cells are inactive, will be given value nan
            * IBOUND<0  coresponding cells have prescribed head
        HI: np.ndarray (nlay, nrow, ncol)
            Initial heads
        FQ: np.ndarray of floats (nlay, nrow, ncol)
            Prescribed cell flows (injection positive)
        """
        
        # dtypes for the boundary types (alike those of flopy)    
        self.dtype = {
            "drn":  np.dtype([('Ig', int), ('h', float), ('C', float)]),
            "riv":  np.dtype([('Ig', int), ('h', float), ('C', float), ('rbot', float)]),
            "ghb":  np.dtype([('Ig', int), ('h', float), ('C', float)]),
            "drn1": np.dtype([('Ig', int), ('phi',   float), ('h', float),
                            ('h0', float), ('N', float), ('w', float)]), # For free runoff
            # phi at N, h at N, and w at N are all for q=N, h0 is ditch bottom elev. w=ditch width
        }
        
        self.gr = gr
        self.K  = K
        self.c = c
        self.IBOUND = IBOUND
        self.HI = HI
        self.FQ = FQ
        
        Nz, Ny, Nx = self.gr.shape

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

        # from this we have the width of columns, rows and layers
        dx = self.gr.dx.reshape(1, 1, Nx)
        dy = self.gr.dy.reshape(1, Ny, 1)
        dz = self.gr.dz.reshape(Nz, 1, 1)

        inact  = (self.IBOUND==0).reshape(self.gr.nod,) # boolean vector denoting inacive cells
        
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
        Rx2 = Rx2.reshape(self.gr.nod,)
        Rx2[inact] = np.inf
        Rx2=Rx2.reshape(self.gr.shape)
        
        Rx1 = Rx1.reshape(self.gr.nod,)
        Rx1[inact] = np.inf
        Rx1=Rx1.reshape(self.gr.shape)
        
        Ry  = Ry.reshape( self.gr.nod,)
        Ry[ inact] = np.inf
        Ry=Ry.reshape(self.gr.shape)
        
        Rz  = Rz.reshape( self.gr.nod,)
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
                        )),(self.gr.nod, self.gr.nod))
        
        # Diagonal as a 1D vector
        self.adiag = np.array(-self.A.sum(axis=1))[:,0]
        self.Cx, self.Cy, self.Cz = Cx, Cy, Cz
        
        # Get storage from input line if None or not None
        if S is None:
            Sy, S = 0.2, 0.001            
            self.Sto    = gr.AREA * S
            self.Sto[0] = gr.Area * Sy
        elif np.all(gr.shape == S.shape):
            self.Sto = S
        elif np.isscalar(S):
            self.Sto = gr.const(S) * gr.AREA
        else:            
            S = np.atleast_1d(S).flatten() 
            if len(S) <= gr.nz:
                self.Sto = gr.const(S[-1])
                self.Sto[:len(S), :, :] = S[:, np.newaxis, np.newaxis]
                self.Sto = gr.const(S)
                self.Sto *= gr.AREA
            else:
                raise ValueError(f"S must have shape {gr.shape} or len nz={gr.nz}")
        return
    
    @staticmethod
    def extend_dtype(data, fields=None):
        """Add fields to the array by extending its dtype.
        
        Parameters
        ----------
        data: np.ndarray with a given dtype
            The original (recarray)
        fields: a list of extra fields
            e.g. [('this', float), ('that', int64)]        
        """
        
        new_dtype =data.dtype.descr + fields
        extended_data = np.zeros(data.shape, dtype=new_dtype)
        for name in data.dtype.names:
            extended_data[name] = data[name]
        return extended_data

        
    def simulate(self, DRN=None, RIV=None, GHB=None, DRN1=None, dt0=None, htol=0.00001, maxiter=50):
        """Compute a 3D steady state finite diff. model

        Parameters
        ----------
        gr: mfgrid.Grid object instance
            object holding the grid/mesh (see mfgrid.Grid)
        K: np.ndarray of floats (nlay, nrow, nocl) or a 3-tuple of such array
            if 3-tuple then the 2 np.ndarrays are kx, ky and kz
                kx  --- array of cell conductivities along x-axis (Ny, Nx, Nz)
                ky  --- same for y direction (if None, then ky=kx )
                kz  --- same for z direction
            else: kx = ky = kz = K
        c: np.ndarray (nlay - 1, nrow, ncol) or None of not used
            Resistance agains vertical flow between the layers [d]
        DRN: array of dtype:
            dtype = dtype([('Ig', 'int'), ('h', float), ('C', float)]) 
        GHB: array ncell of the following dtype:
            dtype = dtype([('Ig', 'int'), ('h', float), ('C', float)])
            Ig = global cell index, h = fixed head and C = conductance
        RIV: array of dtype
            dtype = dtype([('Ig', 'I'), ('h', float), ('C', float)], ('rbot', float)])
            Ig = global index, h = river stage, C = conductance, rbot is river bottom elevation.
        DRN1: array of dtype
            dtype([('Ig', 'int'), ('phi, float), ('h', float), ('h0', float), ('N', float), ('w', float)])
            Ig is global index,
            phi = head at mean recharge N
            h = drainage level at mean recharge N
            h0 = bottom elevation of (deepest) ditches
            N = mean recharge
            w = ditch width at mean recharge N
        dt0: None | float
            Initial time step to stabilize non-linear behavior caused by DRN, RIV and DRN1 boundaries.
            With each outer iteration dt is doubled to make sure we reach the steady state.
            We will make maxouter at least 25, which turns dt0 in 33 million.
        atol, rtol: float (defaults: 0.0 and 1e-5)
            absolute and relative tolerance in sparse.linalg.cg (conjugate gradient) solver.

        Returns
        -------
        out: dict
            a dict with fields Phi, cell flow Q, Qx, Qy, Qz, RIV, DRN, GHB, DRN1
            Output shapes are (Nz, Ny, Nx) for Phi and Q and
            (Nz, Ny,Nx-1) for Qx, (Nz,Ny-1,Nx) for Qy, (Nz-1,Ny,Nx) for Qz, while
            DRN, RIV, GHB, DRN1 have a column that holds the discharge for the the given nodes.

        TO 160905,, 240316, 250320
        """
        # DRN boundaries
        if DRN is not None:
            if not DRN.dtype == self.dtype['drn']:
                raise ValueError(
                    f"""DRN must have dtype:\n{self.dtype['drn']}\nnot\n{DRN.dtype}"""
                )
            DRN = self.__class__.extend_dtype(DRN, fields=[('Q', float), ('phi', float)])
            
        # RIV boundaries
        if RIV is not None:
            if not RIV.dtype == self.dtype['riv']:
                raise ValueError(
                    f"""RIV must have dtype:\n{self.dtype['riv']}\nnot\n{RIV.dtype}"""
                )
            RIV = self.__class__.extend_dtype(RIV, fields=[('Q', float), ('phi', float)])

            
        # General head boundaries
        if GHB is not None:
            if not GHB.dtype == self.dtype['ghb']:
                raise ValueError(
                    f"""GHB must have dtype:\n{self.dtype['ghb']}\nnot\n{GHB.dtype}""")
            GHB = self.__class__.extend_dtype(GHB, fields=[('Q', float), ('phi', float)])
            
        # Free drainage boundaries (kind of drains)
        if DRN1 is not None:
            if not DRN1.dtype == self.dtype['drn1']:
                raise ValueError(
                    f"""DRN1 must have dtype:\n{self.dtype['drn1']}\nnot\n{DRN1.dtype}"""
                )
            DRN1 = self.__class__.extend_dtype(DRN1, fields=[('Q', float), ('C', float), ('c', float),
                                                             ('gamma', float), ('eta', float), ('beta', float), ('ge', float)])      
                
            # Compute the initial coefficients taking phi=phiN, h=hN, q=N
            # Make sure phiN > hN and hN > h0, N>0, so that dphi>0 and y>0 initially
            
        #cell numbers for neighbors
        
        active = (self.IBOUND > 0).reshape(self.gr.nod,)  # boolean vector denoting the active cells
        inact  = (self.IBOUND ==0).reshape(self.gr.nod,) # boolean vector denoting inacive cells
        fxhd   = (self.IBOUND < 0).reshape(self.gr.nod,)  # boolean vector denoting fixed-head cells

        nod = int(self.gr.nod)
        
        HI = self.HI.copy().reshape(nod, 1)
        
        # Set up the initial head vector
        Phi = HI[:, 0].copy()

        # Outer iterations
        dt = dt0 if dt0 is not None else 1.
        for iouter in range(maxiter):
            # Keep A's diagonal as a 1D vector
            adiag = self.adiag.copy()

            # St up up the Right-hand side vector, just using FQ, which is a gr.shape array.
            RHS = self.FQ.copy().reshape(nod, 1)

            if np.any(fxhd):
                RHS -= (self.A + sp.diags(adiag, 0))[:, fxhd] @ HI[fxhd]

            # Add the boundary conditions
            if DRN is not None:
                Ig = DRN['Ig']
                DRN['phi'] = Phi[Ig]
                mask = Phi[Ig] > DRN['h'] # select cells that actually leak
                if True: # Non-linear          
                    adiag[Ig[mask]] += DRN['C'][mask]
                    RHS[Ig[mask], 0] += DRN['C'][mask] * DRN['h'][mask]
                else: # Linear
                    adiag[Ig] += DRN['C']
                    RHS[Ig, 0] += DRN['C'] * DRN['h']

            if RIV is not None:
                Ig =RIV['Ig']
                RIV['phi'] = Phi[Ig]
                if True: # Non-linear
                    mask = Phi[Ig] > RIV['rbot']        
                    adiag[Ig[mask]] += RIV['C'][mask]
                    RHS[Ig[ mask], 0] += RIV['C'][mask] * RIV['h'][mask]
                    RHS[Ig[~mask], 0] += RIV['C'][~mask] * (RIV['h'][~mask] - RIV['rbot'][~mask])
                else: # L inear
                    adiag[Ig] += RIV['C']
                    RHS[Ig, 0] += RIV['C'] * RIV['h']

            if GHB is not None:
                Ig = GHB['Ig']
                GHB['phi'] = Phi[Ig]
                adiag[Ig] += GHB['C']
                RHS[  Ig, 0] += GHB['C'] * GHB['h']

            if DRN1 is not None:                
                Ig = DRN1['Ig']
                if iouter == 0:
                    delta = 0.01
                    lamb = delta * np.exp(1)
                    dphi_h0 = DRN1['phi'] - DRN1['h0']
                    y = DRN1['h'] - DRN1['h0']
                    if not np.all((DRN1['phi'] > DRN1['h']) & (DRN1['h'] > DRN1['h0'])):
                        raise ValueError("Initially we must have DRN1['phi'] > DRN1['h] > DRN1['h0']")
                    if not np.all(DRN1['N'] > 0):
                        raise ValueError("Initially we must have DRN1['N'] > 0")
                    DRN1['c'] = (dphi_h0 - y) / DRN1['N']
                    DRN1['gamma'] = DRN1['c'] * np.sqrt(DRN1['N'])
                    DRN1['eta'  ] = y / np.sqrt(DRN1['N'])
                    DRN1['ge'] = (DRN1['gamma'] + DRN1['eta']) ** 2
                    DRN1['beta']  = (DRN1['w'] / 2) / np.sqrt(y)
                    DRN1['C'] = dphi_h0 * gr.Area.ravel()[Ig] / DRN1['ge']
                else:
                    DRN1['phi'] = Phi[Ig]
                    dphi_h0 = Phi[Ig] - DRN1['h0']
                    mask = dphi_h0 > lamb
                    DRN1['C'][ mask] = dphi_h0[mask]
                    DRN1['C'][~mask] = delta * np.exp(dphi_h0[~mask] / lamb)
                    DRN1['C'] *=  gr.Area.ravel()[Ig] / DRN1['ge']
                    DRN1['C'] = gr.Area.ravel()[Ig] / 200. # Temporarily
                adiag[Ig] += DRN1['C']
                RHS[Ig, 0] += DRN1['C'] * DRN1['h0']
                
            # Quasi time stepping to stabilize non linear behavior

            if np.any(np.array([p is not None for p in [DRN, RIV, DRN1]])).item():
                # If any of the non-linear boundaries are not None, use cg and use outer iterations
                
                adiag += self.Sto.ravel() / dt
                RHS[:, 0] += self.Sto.ravel() / dt * Phi                
                
                Phi[active] = la.spsolve((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active] )                
                # Phi[active], info = la.cg((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active], x0=HI[active], rtol=rtol, atol=atol, maxiter=250)
                info = 1 # temporarily
                err = np.abs(Phi[active] - HI.ravel()[active]).max()
                
                print(f"iouter = {iouter}, info={info}, err={err:.5g} m")                
                if err < htol:
                    print("Converged, normal termination.")
                    break
                elif iouter == maxiter - 1:                    
                    print("Not converged!")
                else:
                    HI[:, 0] = Phi
                    dt = dt / 2                    
            else:
                Phi[active] = la.spsolve((self.A + sp.diags(adiag, 0))[active][:,active], RHS[active] )                
                print("No outer iterations needed.")
                break


        # Prepare output, all items are returned in a single dictionary out
        out=dict()
        out.update(gr=gr)
        
        # reshape Phi to shape of grid
        Phi = Phi.reshape(self.gr.shape)
        out.update(Phi=Phi)

        # Net cell inflow
        Q = self.gr.const(0.)
        Q.ravel()[active]  = (self.A + sp.diags(adiag, 0))[active][:,active] @ Phi.ravel()[active]
        out.update(Q=Q)
        
        #Flows across cell faces
        Qx =  -np.diff(Phi, axis=2) * self.Cx
        Qy =  +np.diff(Phi, axis=1) * self.Cy
        Qz =  +np.diff(Phi, axis=0) * self.Cz
        out.update(Qx=Qx, Qy=Qy, Qz=Qz)
                
        Qfh = (self.A + sp.diags(self.adiag, 0))[:, fxhd] @ Phi.ravel()[fxhd]
    
        out.update(Qfq=FQ, Qfh=Qfh)
        
        if DRN is not None:                      
            mask = Phi.ravel()[DRN['Ig']] > DRN['h']
            DRN['Q'][mask] = DRN['C'][mask] * (DRN['h'][mask] - Phi.ravel()[DRN['Ig'][mask]])
            out.update(DRN=DRN)

        if RIV is not None:            
            mask = Phi.ravel()[RIV['Ig']] > RIV['rbot']
            RIV['Q'][ mask] = RIV['C'][ mask] * (RIV['h'][ mask] - Phi.ravel()[RIV['Ig'][mask]])
            RIV['Q'][~mask] = RIV['C'][~mask] * (RIV['h'][~mask] - RIV['rbot'][~mask])
            out.update(RIV=RIV)

        if GHB is not None:            
            GHB['Q'] = GHB['C'] * (GHB['h'] - Phi.ravel()[GHB['Ig']])
            out.update(GHB=GHB)

        if DRN1 is not None:            
            mask = Phi.ravel()[DRN1['Ig']] > DRN1['h0']
            DRN1['Q'][mask] = DRN1['C'][mask] * (DRN1['h'][mask] - Phi.ravel()[DRN1['Ig'][mask]])
            out.update(DRN1=DRN1)

        # Finally: set inactive cells to np.nan
        out['Phi'][inact.reshape(self.gr.shape)] = np.nan # put np.nan at inactive locations

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
    print("\n===== Water balance of the entire model =====")
    Q1 = out['Q'].sum()
    print(f"Q1    = {Q1:10.2f} m3/d, total internal flow into cells.")
    
    print("Boundary components:")
    Qfh = out['Qfh'].sum()
    print(f"fQfh  = {Qfh:10.2f} m3/d, flow from FH cells.")
    Q2 = Qfh
    
    Qfq = out['Qfq'].sum()
    print(f"Qfq   = {Qfq:10.2f} m3/d, flow from FQ cells.")
    Q2 += Qfq
    
    if 'DRN' in out:
        Qdrn = out['DRN']['Q'].sum()
        print(f"Qdrn  = {Qdrn:10.2f} m3/d, flow form DRN cell.s")
        Q2 += Qdrn
        
    if 'GHB' in out:
        Qghb = out['GHB']['Q'].sum()
        print(f"Qghb  = {Qghb:10.2f} m3/d, flow from GHB cells.")
        Q2 += Qghb
        
    if 'RIV' in out:
        Qriv = out['RIV']['Q'].sum()
        print(f"Qriv  = {Qriv:10.2f} m3/d, flow from RIV cells.")
        Q2 += Qriv
        
    if 'DRN1' in out:
        Qdrn1 = out['DRN1']['Q'].sum()
        print(f"Qdrn1 = {Qdrn1:10.2f} m3/d, flow from DRN1 cells.")
        Q2 += Qdrn1
    
    print()
    print("Total waterbalance for model")
    print(f"Q1 (internal)    = {Q1:10.2f} m3/d, total internal flow")
    print(f"Q2 (boundaries)  = {Q1:10.2f} m3/d, total from bounaries")
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
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, c=c, IBOUND=IBOUND)
    
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
    B  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    FQ = gr.const(0.); FQ[-1, 0, 0] = kw['Q']   # m3/d fixed flows
    HI = gr.const(0.)                           # m, initial heads
    
    IBOUND = gr.const(1); IBOUND[0, :, :] = -1  # modflow like boundary array
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, c=c, IBOUND=IBOUND) # run model
    
    ax = newfig(kw['title'], 'r [m]', 's [m]', xscale='log', xlim=[1e-3, r[-1]])
    
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * spec.k0(gr.xm / B) / (kw['rw']/ B * spec.k1(kw['rw']/ B)), 'b-',label='analytic')
    ax.legend()
    return out
    
    
def deGlee_GHB(kw):
    """Run Axial symmetric example, as before, but now using GHB instead of an extra layer on top.
    """
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    B  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))

    z = (kw['z0'] - np.cumsum(np.hstack((0., kw['D']))))[1:] # no top layer
    
    gr = mfgrid.Grid(r, None, z, axial=True)   # generate grid
    
    FQ = gr.const(0.); FQ[-1, 0, 0] = kw['Q']     # m3/d fixed flows
    HI = gr.const(0.)                       # m, initial heads
    
    IBOUND = gr.const(1, dtype=int)         # modflow like boundary array
    
    k = gr.const(kw['k'][-1])                         # full 3D array of conductivities
    
    cells = np.zeros(gr.shape, dtype=bool); cells.ravel()[gr.NOD[0]] = True
    hds = HI[cells]
    C   = gr.Area.ravel() / float(kw['c'])
    GHB = gr.GHB(cells, hds, C)
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB) # run model
    
    ax = newfig(kw['title'] + ' (Using GHB)', 'r [m]', 's [m]', xscale='log', xlim=[1e-3, r[-1]])
    
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * spec.k0(gr.xm / B) / (kw['rw']/ B * spec.k1(kw['rw']/ B)), 'b-',label='analytic')
    ax.legend()
    return out

def freeDrainage(kw):
    """Run Axial symmetric example, as before, but now using GHB instead of an extra layer on top.
    """
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    B  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))

    z = (kw['z0'] - np.cumsum(np.hstack((0., kw['D']))))[1:] # no top layer
    
    gr = mfgrid.Grid(r, None, z, axial=True)   # generate grid
    
    FQ = gr.const(0.); FQ[-1, 0, 0] = kw['Q']     # m3/d fixed flows
    HI = gr.const(0.)                       # m, initial heads
    
    IBOUND = gr.const(1, dtype=int)         # modflow like boundary array
    
    k = gr.const(kw['k'][-1])                         # full 3D array of conductivities
    
    cells = np.zeros(gr.shape, dtype=bool); cells.ravel()[gr.NOD[0]] = True
    hds = HI[cells]
    C   = gr.Area.ravel() / float(kw['c'])
    GHB = gr.GHB(cells, hds, C)
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB) # run model
    
    ax = newfig(kw['title'] + ' (Using GHB)', 'r [m]', 's [m]', xscale='log', xlim=[1e-3, r[-1]])
    
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * spec.k0(gr.xm / B) / (kw['rw']/ B * spec.k1(kw['rw']/ B)), 'b-',label='analytic')
    ax.legend()
    return out

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
    'DeGlee': {
        'title': 'Geglee axial symmetric flow',
        'comment': """Axial symmetric example, well in semi-confined aquifer (De Glee case)
            De Glee was a Dutch engineer/groundwater hydrologist and later the
            first director of the water company of the province of Groningen.
            His PhD (1930) solved the axial symmetric steady state flow to a well
            in a semi confined aquifer using the Besselfunctions of the second kind,
            known as K0 and K1.
            The example computes the heads in the regional aquifer below a semi confining
            layer with a fixed head above. It uses two model layers a confining one in
            which the heads are fixed and a semi-confined aquifer with a prescribed
            extraction at r=rw. If rw>>0, both K0 and K1 Bessel functions are needed.
            The grid is signaled to use inteprete the grid as axially symmetric.
            """,
        'z0': 0.,
        'Q': -1200.,
        'rw':   .25,
        'D' : np.array([10., 50.]),
        'c' :  np.array([[250.]]),
        'k' :  np.array([np.inf, 10]),  # m/d conductivity of regional aquifer
        'r' : np.logspace(-2, 4, 61),  # distance to well center
    },
    'FreeDrainage': {
        'title': 'Free drainage axial symmetric flow',
        'comment': """fill in later
            """,
        'z0': 0.,
        'Q': -1200.,
        'rw':   .25,
        'D' : np.array([10., 50.]),
        'c' :  np.array([[250.]]),
        'k' :  np.array([np.inf, 10]),  # m/d conductivity of regional aquifer
        'r' : np.logspace(-2, 4, 61),  # distance to well center
    },
}  

if __name__ == "__main__":
    # out1 = mazure(cases['Mazure'])
    # out2 = deGlee(cases['DeGlee'])
    # out2 = deGlee_GHB(cases['DeGlee'])
    # out3 = freeDrainage(cases['FreeDrainage'])
    
    r = np.hstack((0., np.logspace(-1., np.log10(5000.), 25)))
    r = np.unique(np.hstack((r, r[-1] - r)))
    r = np.linspace(0, 5000., 101)
    
    z = np.array([1., -0., -50.])
    # gr = mfgrid.Grid(r, None, z, axial=True)
    gr = mfgrid.Grid(r, None, z, axial=False)
    kx, kz = 10., 1000.
    K = (gr.const(kx), gr.const(kx), gr.const(kz))
    c = gr.const(100.)[0:1]
    IBOUND = gr.const(1, dtype=int)
    HI = gr.const(0.)
    FQ = gr.const(0.)
    # Q = -1200.
    Q = -10.0
    Q = 0.
    N = 0.002
    FQ[-1, 0, 0] = Q
    ilay = 1

    variants= {'Glee', 'ccc', 'DRN', 'GHB', 'RIV', 'DRN1'}
    variants= {'Glee', 'ccc', 'DRN', 'GHB', 'RIV'}
    # variants= {'Glee', 'ccc', 'DRN', 'GHB'}
    variants= {'ccc', 'DRN', 'GHB', 'RIV', 'DRN1'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title="Fdm3", xlabel="r [m]", ylabel="head [m]", xscale="linear") # "log")
    
    # Basic, De Glee
    if 'Glee' in variants:
        dGlee = wf.wDeGlee(xw=0., yw=0., rw=gr.xm[0], z1=None, z2=None, aqprops={'kD':kx * gr.dz[-1], 'c':c[0, 0, 0]})
        ddGl = dGlee.dd(x=gr.xm, Q=Q)
        ax.plot(gr.xm, ddGl, '.', label='De Glee')

    # Basic, using c (resistance
    if 'ccc' in variants:
        IBOUND = gr.const(1, dtype=int)
        IBOUND[0] = -1
        # IBOUND[-1, 0, -1] = -1
        HI = gr.const(0)
        HI[-1, 0, -1] = 1.0
        FQ = gr.const(0.)
        FQ[-1, 0, 0] = Q
        FQ[-1] += gr.Area * N
        mdl_ccc = Fdm3(gr=gr, K=K, c=c, IBOUND=IBOUND, HI=HI, FQ=FQ)
        out_ccc = mdl_ccc.simulate()
        ax.plot(gr.xm, out_ccc['Phi'][ilay][0], '-.',  label=f"ccc, Phi[{ilay}]")
        
    # Using GHB (should be the same as De Glee or ccc)
    if 'GHB' in variants: 
        IBOUND = gr.const(1, dtype=int)
        # IBOUND[-1, 0, -1] = -1
        HI = gr.const(0.)
        HI[-1, 0, -1] = 1.0
        FQ = gr.const(0.)
        FQ[-1, 0, 0] = Q
        FQ[-1] += gr.Area * N  
        mdl_ghb = Fdm3(gr=gr, K=K, c=None, IBOUND=IBOUND, HI=HI, FQ=FQ)
    
        GHB = np.zeros(gr.nx, dtype=mdl_ghb.dtype['ghb'])
        GHB['Ig'] = gr.NOD[0, 0, :]
        GHB['C'] = gr.AREA[0, 0, :] /  c[0, 0, :]
        GHB['h'] = HI[0, 0, :]

        out_ghb = mdl_ghb.simulate(GHB=GHB)
        ax.plot(gr.xm, out_ghb['Phi'][ilay][0], '-', label=f"GHB, Phi[{ilay}]")
    
    # Using DRN (should be the same Blom)
    if 'DRN' in variants:
        IBOUND = gr.const(1, dtype=int)
        # IBOUND[-1, 0, -1] = -1         
        FQ = gr.const(0.)
        FQ[-1, 0, 0] = Q
        FQ[-1] += gr.Area * N
        
        HI = gr.const(0.)
        HI[-1, 0, -1] = 1.0
        mdl_drn = Fdm3(gr=gr, K=K, c=None, IBOUND=IBOUND, HI=HI, FQ=FQ)

        DRN = np.zeros(gr.nx, dtype=mdl_drn.dtype['drn'])
        DRN['Ig'] = gr.NOD[0, 0, :]
        DRN['C']  = gr.Area[0, :] /  c[0, 0, :]
        DRN['h']  = HI[0, 0, :]

        out_drn = mdl_drn.simulate(DRN=DRN)
        ax.plot(gr.xm, out_drn['Phi'][ilay][0], '-', label=f"DRN, Phi[{ilay}]")

    # Using RIV (river)
    if 'RIV' in variants:
        IBOUND = gr.const(1, dtype=int)
        # IBOUND[-1, 0, -1] = -1   
        HI = gr.const(0.)
        HI[-1, 0, -1] = 1.0     
        FQ = gr.const(0.)        
        FQ[-1, 0, 0] = Q
        FQ[-1] += gr.Area * N   
        mdl_riv = Fdm3(gr=gr, K=K, c=None, IBOUND=IBOUND, HI=HI, FQ=FQ)

        RIV = np.zeros(gr.nx, dtype=mdl_riv.dtype['riv'])
        RIV['Ig'] = gr.NOD[0, 0, :]
        RIV['C'] = gr.Area[0, :] /  c[0, 0, :]
        RIV['h']   = HI[0, 0, :]
        RIV['rbot']  = RIV['h']

        out_riv = mdl_riv.simulate(RIV=RIV)
        ax.plot(gr.xm, out_riv['Phi'][ilay][0], label=f"RIV, Phi[{ilay}]")


    # Using DRN1 (free runoff)
    if 'DRN1' in variants:
        IBOUND = gr.const(1, dtype=int)
        # IBOUND[-1, 0, -1] = -1
        HI = gr.const(0.)
        HI[-1, 0, -1] = 1.0       
        FQ = gr.const(0.)
        FQ[-1, 0, 0] = Q
        FQ[-1] += gr.Area * N        
        mdl_drn1 = Fdm3(gr=gr, K=K, c=None, IBOUND=IBOUND, HI=HI, FQ=FQ)

        DRN1 = np.zeros(gr.nx, dtype=mdl_drn1.dtype['drn1'])
        DRN1['Ig'] = gr.NOD[0, 0, :]
        DRN1['phi'] = 0.4
        DRN1['h']   = 0.0
        DRN1['h0']  = -1.0
        DRN1['w'] = 1.0
        DRN1['N']   = N

        out_drn1 = mdl_drn1.simulate(DRN1=DRN1)
        ax.plot(gr.xm, out_drn1['Phi'][ilay][0], label=f"DRN1, Phi[{ilay}]")
        
        ax.plot(gr.xm, DRN1['phi'], '--', label='DRN1-phi')
        ax.plot(gr.xm, DRN1['h'], '-.', label='DRN1-h')
        ax.plot(gr.xm, DRN1['h0'], '--', label='DRN1-h0')   
    
    ax.grid()
    ax.legend()
    
    watbal(out_ccc)
    watbal(out_ghb)
    watbal(out_drn)
    watbal(out_riv)
    watbal(out_drn1)
        
    print('Done')
    plt.show()
    
