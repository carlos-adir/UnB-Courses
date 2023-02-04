import numpy as np
from compmec import nurbs
from typing import Callable, Tuple
from helper import getD, getH, solve_system, plot_field
from matplotlib import pyplot as plt

def fit_spline_curve(N: nurbs.SplineBaseFunction, f: Callable[[float], float]) -> np.array:
    tsample = np.linspace(0, 1, 4*N.npts)
    Lt = N(tsample)
    ctrlpoints = np.linalg.lstsq(Lt.T, f(tsample), rcond=None)[0]
    return ctrlpoints

def fit_spline_surface(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, f: Callable[[float, float], float]) -> np.array:
    xsample = np.linspace(0, 1, 4*Nx.npts)
    ysample = np.linspace(0, 1, 4*Ny.npts)
    Lx = Nx(xsample)
    Ly = Ny(ysample)
    Kx = np.linalg.inv(Lx @ Lx.T) @ Lx
    Ky = np.linalg.inv(Ly @ Ly.T) @ Ly
    F = np.zeros((len(xsample), len(ysample)), dtype="float64")
    for i, xi in enumerate(xsample):
        for j, yj in enumerate(ysample):
            F[i, j] = f(xi, yj)
    return Kx @ F @ Ky.T


def compute_pressure_from_current_line(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S: np.ndarray)->np.ndarray:
    px, nx, Ux = Nx.degree, Nx.npts, Nx.knotvector
    py, ny, Uy = Ny.degree, Ny.npts, Ny.knotvector
    Dpx = getD(px, Ux)
    Dpy = getD(py, Uy)
    Mat2X = np.einsum("jb,kc,ibc->ijk", Dpx, Dpx, getH(Nx, px, px-1, px-1))
    Mat2Y = np.einsum("jb,kc,ibc->ijk", Dpy, Dpy, getH(Ny, py, py-1, py-1))
    
    Mat1X = np.einsum("i,j,k->ijk", Nx(1), Nx(1), Dpx @ Nx[:, px-1](1))
    Mat1X -= np.einsum("i,j,k->ijk", Nx(0), Nx(0), Dpx @ Nx[:, px-1](0))
    Mat1X -= Mat2X
    Mat1X -= np.einsum("jik->ijk", Mat2X)
    
    Mat1Y = np.einsum("i,j,k->ijk", Ny(1), Ny(1), Dpy @ Ny[:, py-1](1))
    Mat1Y -= np.einsum("i,j,k->ijk", Ny(0), Ny(0), Dpy @ Ny[:, py-1](0))
    Mat1Y -= Mat2Y
    Mat1Y -= np.einsum("jik->ijk", Mat2Y)

    Bmat = np.einsum("iac,jbd,ab,cd->ij", Mat1X, Mat1Y, S, S)
    Bmat -= np.einsum("iac,jbd,ab,cd->ij", Mat2X, Mat2Y, S, S)
    Bmat *= 2

    Xb = np.tensordot(Nx[:, px](1), Nx[:, px-1](0), axes=0)
    Xb -= np.tensordot(Nx[:, px](0), Nx[:, px-1](0), axes=0)
    Yb = np.tensordot(Ny[:, py](1), Ny[:, py-1](0), axes=0)
    Yb -= np.tensordot(Ny[:, py](0), Ny[:, py-1](0), axes=0)

    MatXLap = Xb @ Dpx.T - Dpx @ getH(Nx, px-1, px-1) @ Dpx.T
    MatYLap = Yb @ Dpy.T - Dpy @ getH(Ny, py-1, py-1) @ Dpy.T
    Amat = np.tensordot(MatXLap, getH(Ny, py, py), axes=0)
    Amat += np.tensordot(getH(Nx, px, px), MatYLap, axes=0)

    Pbound = np.empty((nx, ny), dtype="float64")
    Pbound.fill(np.nan)
    Pbound[0, 0] = 0  # Para existir uma referencia
    # Pbound[0, ny-1] = 0
    # Pbound[nx-1, 0] = 0
    # Pbound[nx-1, ny-1] = 0

    # Apenas para testar
    Bmat[0, :].fill(0)
    Bmat[nx-1, :].fill(0)
    Bmat[:, 0].fill(0)
    Bmat[:, ny-1].fill(0)
    for j in range(1, ny-1):  # BC at left
        Amat[0, :, j, :].fill(0)
        Amat[0, :, j, j] = Dpx @ Nx[:, px-1](0)
        # Bmat[0, j] = mu * np.einsum("", Dpx, Dpx1, Dpy, Nx(0), S)
    for j in range(1, ny-1):  # BC at right
        Amat[nx-1, :, j, :].fill(0)
        Amat[nx-1, :, j, j] = Dpx @ Nx[:, px-1](1)
    for i in range(1, nx-1):  # BC at lower
        Amat[i, :, 0, :].fill(0)
        Amat[i, i, 0, :] = Dpy @ Ny[:, py-1](0)
    for i in range(1, nx-1):  # BC at upper
        Amat[i, :, ny-1, :].fill(0)
        Amat[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)

    P, _ = solve_system(Amat, Bmat, Pbound)
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



def alpha(U: Tuple[float], i: int, j: int):
    if U[i+j] == U[i]:
        return 0
    return j/(U[i+j]-U[i])


def plot_all_fields(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S = None, U = None, V = None, W = None, P=None):
    xplot = np.linspace(0, 1, 257)
    yplot = np.linspace(0, 1, 257)
    Lx = Nx(xplot)
    Ly = Ny(yplot)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    if S is not None:
        zplot = Lx.T @ S @ Ly
        plot_field(xplot, yplot, zplot, contour=True, ax=axes[0])
    if U is not None:
        zplot = Lx.T @ U @ Ly
        plot_field(xplot, yplot, zplot, contour=True, ax=axes[1])
    if V is not None:
        zplot = Lx.T @ V @ Ly
        plot_field(xplot, yplot, zplot, contour=True, ax=axes[2])
    if W is not None:
        zplot = Lx.T @ W @ Ly
        plot_field(xplot, yplot, zplot, contour=True, ax=axes[3])
    if P is not None:
        zplot = Lx.T @ P @ Ly
        plot_field(xplot, yplot, zplot, contour=True, ax=axes[4])
    axes[0].set_title(r"Linha de corrente $S$")
    axes[1].set_title(r"Horizontal speed $u$")
    axes[2].set_title(r"Vertical speed $v$")
    axes[3].set_title(r"Vorticidade $W$")
    axes[4].set_title(r"Pressure $p$")
    for i in range(5):
        # axes[i].set_xlabel(r"Dimensao $x$")
        # axes[i].set_ylabel(r"Dimensao $y$")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        [axes[i].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Nx.knotvector))[1:-1]]
        [axes[i].axhline(y=yj, ls="dotted", color="k") for yj in list(set(Ny.knotvector))[1:-1]]
