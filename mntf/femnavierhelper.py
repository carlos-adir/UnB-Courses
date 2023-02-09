import numpy as np
from compmec import nurbs
from typing import Callable, Tuple
from helper import getD, getH, solve_system, plot_field
from matplotlib import pyplot as plt

class Fit:

    @staticmethod
    def spline_curve(N: nurbs.SplineBaseFunction, f: Callable[[float], float], BCvals: np.ndarray=None) -> np.array:
        tsample = []
        knots = N.knotvector.knots
        for ta, tb in zip(knots[:-1], knots[1:]):
            tsample.extend(np.linspace(ta, tb, 5, endpoint=False))
        tsample.append(1)
        tsample = np.array(tsample)
        Lt = N(tsample)
        F = np.array([f(ti) for ti in tsample], dtype="float64")
        if BCvals is None:
            return np.linalg.lstsq(Lt.T, F, rcond=None)[0]
        mask = np.isnan(BCvals)
        BCvals[mask] = 0
        F -= Lt.T @ BCvals
        BCvals[mask] = 0
        nunkknown = np.sum(mask)
        indexs = np.zeros((nunkknown, 1), dtype="int32")
        k = 0
        for i in range(N.npts):
            if mask[i]:
                indexs[k, 0] = i
                k += 1
        
        B = np.zeros(nunkknown, dtype="float64")
        A = np.zeros((nunkknown, nunkknown), dtype="float64")
        for ka, (ia, ) in enumerate(indexs):
            B[ka] = Lt[ia] @ F
            for kb, (ib, ) in enumerate(indexs):
                A[ka, kb] = Lt[ia] @ Lt[ib]
        solution = np.linalg.solve(A, B)
        finalresult = np.copy(BCvals)
        for a, (ia, ) in enumerate(indexs):
            finalresult[ia] = solution[a]
        BCvals[mask].fill(np.nan)
        return finalresult

    @staticmethod
    def spline_surface(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, f: Callable[[float, float], float], BCvals: np.ndarray = None) -> np.array:
        nx, ny = Nx.npts, Ny.npts
        xsample = []
        ysample = []
        xknots = Nx.knotvector.knots
        yknots = Ny.knotvector.knots
        for ta, tb in zip(xknots[:-1], xknots[1:]):
            xsample.extend(np.linspace(ta, tb, 5, endpoint=False))
        for ta, tb in zip(yknots[:-1], yknots[1:]):
            ysample.extend(np.linspace(ta, tb, 5, endpoint=False))
        xsample.append(1)
        xsample = np.array(xsample)
        ysample.append(1)
        ysample = np.array(ysample)

        nxs, nys = len(xsample), len(ysample)
        F = np.zeros((nxs, nys), dtype="float64")
        for i, xi in enumerate(xsample):
            for j, yj in enumerate(ysample):
                F[i, j] = f(xi, yj)
        
        Lx = Nx(xsample)
        Ly = Ny(ysample)
        Kx = np.linalg.inv(Lx @ Lx.T) @ Lx
        Ky = np.linalg.inv(Ly @ Ly.T) @ Ly
        if BCvals is None:
            return Kx @ F @ Ky.T
        mask = np.isnan(BCvals)
        BCvals[mask] = 0
        F -= Lx.T @ BCvals @ Ly
        BCvals[mask] = np.nan
        nunkknown = np.sum(mask)
        indexs = np.zeros((nunkknown, 2), dtype="int32")
        k = 0
        for i in range(nx):
            for j in range(ny):
                if mask[i, j]:
                    indexs[k] = i, j
                    k += 1
        B = np.zeros(nunkknown, dtype="float64")
        A = np.zeros((nunkknown, nunkknown), dtype="float64")
        for ka, (ia, ja) in enumerate(indexs):
            B[ka] = Lx[ia] @ F @ Ly[ja].T
            for kb, (ib, jb) in enumerate(indexs):
                A[ka, kb] = (Lx[ia] @ Lx[ib]) * (Ly[ja] @ Ly[jb])
        solution = np.linalg.solve(A, B)
        finalresult = np.copy(BCvals)
        for a, (ia, ja) in enumerate(indexs):
            finalresult[ia, ja] = solution[a]
        BCvals[mask].fill(np.nan)
        return finalresult

    


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


def plot_all_fields(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S):
    xplot = np.linspace(0, 1, 1025)
    yplot = np.linspace(0, 1, 1025)
    px, nx = Nx.degree, Nx.npts
    py, ny = Ny.degree, Ny.npts
    Dpx = getD(px, Nx.knotvector)
    Dpx1 = getD(px-1, Nx.knotvector)
    Dpy = getD(py, Ny.knotvector)
    Dpy1 = getD(py-1, Ny.knotvector)
    Lx = Nx[:, px](xplot)
    dLx = Dpx @ Nx[:, px-1](xplot)
    ddLx = Dpx @ Dpx1 @ Nx[:, px-2](xplot)
    Ly = Ny[:, py](yplot)
    dLy = Dpy @ Ny[:, py-1](yplot)
    ddLy = Dpy @ Dpy1 @ Ny[:, py-2](yplot)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    splot = Lx.T @ S @ Ly
    plot_field(xplot, yplot, splot, contour=True, ax=axes[0])
    uplot = Lx.T @ S @ dLy  # U
    plot_field(xplot, yplot, uplot, contour=True, ax=axes[1])
    vplot = dLx.T @ S @ dLy  # V
    plot_field(xplot, yplot, vplot, contour=True, ax=axes[2])
    wplot = -(ddLx.T @ S @ Ly + Lx.T @ S @ ddLy)  # W
    plot_field(xplot, yplot, wplot, contour=True, ax=axes[3])
    # zplot = Lx.T @ P @ Ly
    # plot_field(xplot, yplot, zplot, contour=True, ax=axes[4])
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
        # [axes[i].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Nx.knotvector))[1:-1]]
        # [axes[i].axhline(y=yj, ls="dotted", color="k") for yj in list(set(Ny.knotvector))[1:-1]]


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