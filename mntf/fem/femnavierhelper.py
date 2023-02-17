import numpy as np
from compmec import nurbs
from typing import Callable, Tuple
from helpernurbs import getD, getH
from helperlinalg import solve_system


    


def compute_pressure_from_current_line(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S: np.ndarray)->np.ndarray:
    print("    Computing pressue")
    px, nx, Ux = Nx.degree, Nx.npts, Nx.knotvector
    py, ny, Uy = Ny.degree, Ny.npts, Ny.knotvector
    Dpx = getD(px, Ux)
    Dpy = getD(py, Uy)

    AsysP = np.tensordot(getH(Nx, 0, 2), getH(Ny, 0, 0), axes=0)
    AsysP += np.tensordot(getH(Nx, 0, 0), getH(Ny, 0, 2), axes=0)
    Hx020 = getH(Nx, 0, 2, 0)
    Hy002 = getH(Ny, 0, 0, 2)
    Hx011 = getH(Nx, 0, 1, 1)
    Hy011 = getH(Ny, 0, 1, 1)
    BsysP = np.einsum("iac,jbd,ab,cd->ij", Hx020, Hy002, S, S)
    BsysP -= np.einsum("iac,jbd,ab,cd->ij", Hx011, Hy011, S, S)
    
    
    Pbound = np.empty((nx, ny), dtype="float64")
    Pbound.fill(np.nan)
    Pbound[0, 0] = 0  # Para existir uma referencia
    # Pbound[0, ny-1] = 0
    # Pbound[nx-1, 0] = 0
    # Pbound[nx-1, ny-1] = 0

    # Apenas para testar
    BsysP[0, :].fill(0)
    BsysP[nx-1, :].fill(0)
    BsysP[:, 0].fill(0)
    BsysP[:, ny-1].fill(0)
    for j in range(1, ny-1):  # BC at left
        AsysP[0, :, j, :].fill(0)
        AsysP[0, :, j, j] = Dpx @ Nx[:, px-1](0)
        # Bmat[0, j] = mu * np.einsum("", Dpx, Dpx1, Dpy, Nx(0), S)
    for j in range(1, ny-1):  # BC at right
        AsysP[nx-1, :, j, :].fill(0)
        AsysP[nx-1, :, j, j] = Dpx @ Nx[:, px-1](1)
    for i in range(1, nx-1):  # BC at lower
        AsysP[i, :, 0, :].fill(0)
        AsysP[i, i, 0, :] = Dpy @ Ny[:, py-1](0)
    for i in range(1, nx-1):  # BC at upper
        AsysP[i, :, ny-1, :].fill(0)
        AsysP[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)

    print("    Solving the system!")
    P, _ = solve_system(AsysP, BsysP, Pbound)
    print("    System solved")
    return P

def compute_U_from_current_line(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S: np.ndarray)->np.ndarray:
    """
    u = partial psi/partial y
    """
    nx = Nx.npts
    py, ny, Uy = Ny.degree, Ny.npts, Ny.knotvector
    Mat = np.linalg.solve(getH(Ny, py, py), getH(Ny, py, py-1) @ getD(py, Uy).T)
    U = np.zeros((nx, ny), dtype="float64")
    for i in range(nx):
        U[i, :] = Mat @ S[i, :]
    return U

def compute_V_from_current_line(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S: np.ndarray)->np.ndarray:
    """
    v = -partial psi/partial x
    """
    px, nx, Ux = Nx.degree, Nx.npts, Nx.knotvector
    ny = Ny.npts
    Mat = np.linalg.solve(getH(Nx, px, px), getH(Nx, px, px-1) @ getD(px, Ux).T)
    V = np.zeros((nx, ny), dtype="float64")
    for j in range(ny):
        V[:, j] = Mat @ S[:, j]
    return V


if __name__ == "__main__":

    px, py = np.random.randint(1, 4, size=(2, ))
    nx, ny = np.random.randint(max(px, py)+1, 7, size=(2,) )
    Pgood = 2*np.random.rand(nx, ny)-1
    Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
    Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
    Nx = nurbs.SplineBaseFunction(Ux)
    Ny = nurbs.SplineBaseFunction(Uy)
    
    f = lambda x, y: Nx(x).T @ Pgood @ Ny(y)

    Pbound = np.copy(Pgood)
    nunknown = np.random.randint(1, nx*ny+1)
    while np.sum(np.isnan(Pbound)) < nunknown:
        i = np.random.randint(0, nx)
        j = np.random.randint(0, ny)
        Pbound[i, j] = np.nan
    Ptest = Fit.spline_surface(Nx, Ny, f, Pbound)
    print("Pgood = ")
    print(Pgood)
    print("Pbound = ")
    print(Pbound)
    print("Ptest = ")
    print(Ptest)
    np.testing.assert_almost_equal(Ptest, Pgood)