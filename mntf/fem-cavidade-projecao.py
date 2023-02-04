import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from compmec import nurbs
from helper import getD, getH, plot_field, solve_system
from typing import Tuple, Callable
from femnavierhelper import *
np.set_printoptions(precision=2, suppress=True)

# Analitico
xsymb, ysymb = sp.symbols("x y", real=True)
# Ssymb = -(sp.sin(sp.pi * xsymb)*sp.sin(sp.pi*ysymb))**2  # Linhas de corrente
Ssymb = -(xsymb*(1-xsymb)) * ysymb *(1-ysymb)
Usymb = sp.diff(Ssymb, ysymb)
Vsymb = sp.diff(-Ssymb, xsymb)
Wsymb = sp.diff(Vsymb, xsymb) - sp.diff(Usymb, ysymb)
laplP = 2 * (sp.diff(Ssymb, (xsymb, 2)) * sp.diff(Ssymb, (ysymb, 2)) - sp.diff(Ssymb, xsymb, ysymb))
laplP = sp.expand(laplP)
laplP = sp.simplify(laplP)
laplP = laplP.factor(laplP)
Psymb = sp.sympify(0)
print("Ssymb = ", sp.simplify(sp.expand(Ssymb)))
print("Usymb = ", sp.simplify(sp.expand(Usymb)))
print("Vsymb = ", sp.simplify(sp.expand(Vsymb)))
print("Wsymb = ", sp.simplify(sp.expand(Wsymb)))
print("laplP = ", sp.simplify(sp.expand(laplP)))
print("Psymb = ", sp.simplify(sp.expand(Psymb)))
Sinit = sp.lambdify((xsymb, ysymb), Ssymb)
Uinit = sp.lambdify((xsymb, ysymb), Usymb)
Vinit = sp.lambdify((xsymb, ysymb), Vsymb)
Winit = sp.lambdify((xsymb, ysymb), Wsymb)
Pinit = sp.lambdify((xsymb, ysymb), Psymb)
print("Uu = ", sp.simplify(sp.expand(Usymb.subs(ysymb, 1))))
print("Ub = ", sp.simplify(sp.expand(Usymb.subs(ysymb, 0))))
print("Ul = ", sp.simplify(sp.expand(Usymb.subs(xsymb, 0))))
print("Ur = ", sp.simplify(sp.expand(Usymb.subs(xsymb, 1))))
print("Vu = ", sp.simplify(sp.expand(Vsymb.subs(ysymb, 1))))
print("Vb = ", sp.simplify(sp.expand(Vsymb.subs(ysymb, 0))))
print("Vl = ", sp.simplify(sp.expand(Vsymb.subs(xsymb, 0))))
print("Vr = ", sp.simplify(sp.expand(Vsymb.subs(xsymb, 1))))


@np.vectorize
def upper_speed(x: float, t: float) -> float: 
    """
    Boundary condition of speed
    u(x, y=1) = upper_speed(x, t)
    """
    value = np.sin(np.pi*x)**2
    tofset = 0.1
    if t > tofset:
        return value
    return t*value/tofset
    
def fit_spline_curve(N: nurbs.SplineBaseFunction, f: Callable[[float], float]) -> np.array:
    tsample = np.linspace(0, 1, 129)
    Lt = N(tsample)
    ctrlpoints = np.linalg.lstsq(Lt.T, f(tsample), rcond=None)[0]
    return ctrlpoints

def fit_spline_surface(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, f: Callable[[float, float], float]) -> np.array:
    xsample = np.linspace(0, 1, 129)
    ysample = np.linspace(0, 1, 129)
    Lx = Nx(xsample)
    Ly = Ny(ysample)
    Kx = np.linalg.inv(Lx @ Lx.T) @ Lx
    Ky = np.linalg.inv(Ly @ Ly.T) @ Ly
    F = np.zeros((len(xsample), len(ysample)), dtype="float64")
    for i, xi in enumerate(xsample):
        for j, yj in enumerate(ysample):
            F[i, j] = f(xi, yj)
    return Kx @ F @ Ky.T

@np.vectorize
def s_initial(x: float, y: float):
    return Sinit(x, y)

@np.vectorize
def u_initial(x: float, y: float):
    return Uinit(x, y)

@np.vectorize
def v_initial(x: float, y: float):
    return Vinit(x, y)

@np.vectorize
def w_initial(x: float, y: float):
    return Winit(x, y)

def compute_pressure_from_current_line(Nx, Ny, S: np.ndarray)->np.ndarray:
    px, Ux = Nx.degree, Nx.knotvector
    py, Uy = Ny.degree, Ny.knotvector
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
    for i in range(1, nx-1):  # BC at left
        Amat[i, :, 0, :].fill(0)
        Amat[i, i, 0, :] = Dpy @ Ny[:, py-1](0)
    for i in range(1, nx-1):  # BC at right
        Amat[i, :, ny-1, :].fill(0)
        Amat[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)
    

    P, _ = solve_system(Amat, Bmat, Pbound)
    return P

    
    

nx, ny = 7, 7
px, py = 3, 3
tmax, nt = 3, 10001
dt = tmax/(nt-1)
Re = 10
mu = 1/Re


Ux = nurbs.GeneratorKnotVector.uniform(degree=px, npts=nx)
Uy = nurbs.GeneratorKnotVector.uniform(degree=py, npts=ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)

xsample = np.linspace(0, 1, 2*nx)
ysample = np.linspace(0, 1, 2*ny)
Lx = Nx(xsample)
Ly = Ny(ysample)

U = np.zeros((nt, nx, ny), dtype="float64")  # Horiziontal speed
V = np.zeros((nt, nx, ny), dtype="float64")  # Vertical speed
P = np.zeros((nt, nx, ny), dtype="float64")  # Pressure
W = np.zeros((nt, nx, ny), dtype="float64")  # Vorticity
S = np.zeros((nt, nx, ny), dtype="float64")  # Linhas de corrente

S[0] = fit_spline_surface(Nx, Ny, s_initial)
U[0] = fit_spline_surface(Nx, Ny, u_initial)
V[0] = fit_spline_surface(Nx, Ny, v_initial)
W[0] = fit_spline_surface(Nx, Ny, w_initial)
P[0] = compute_pressure_from_current_line(Nx, Ny, S[0])

Hpxpx = getH(Nx, px, px)
Hpypy = getH(Ny, py, py)
Hpx1px1 = getH(Nx, px-1, px-1)
Hpy1py1 = getH(Ny, py-1, py-1)
Dpx = getD(px, Ux)
Dpy = getD(py, Uy)
Hpxpxpx = getH(Nx, px, px, px)
Hpypypy = getH(Ny, py, py, py)

Xb = np.tensordot(Nx[:, px](1), Nx[:, px-1](0), axes=0)
Xb -= np.tensordot(Nx[:, px](0), Nx[:, px-1](0), axes=0)
Yb = np.tensordot(Ny[:, py](1), Ny[:, py-1](0), axes=0)
Yb -= np.tensordot(Ny[:, py](0), Ny[:, py-1](0), axes=0)

MatXLap = Xb @ Dpx.T - Dpx @ Hpx1px1 @ Dpx.T
MatYLap = Yb @ Dpy.T - Dpy @ Hpy1py1 @ Dpy.T

Mat1PosX = np.einsum("jy,kz,iyz->ijk", Dpx, Dpx, getH(Nx, px,px-1,px-1))
Mat2PosX = np.einsum("jy,iyk->ijk", Dpx, getH(Nx, px, px-1, px))
Mat2PosY = np.einsum("kz,ijz->ijk", Dpy, getH(Ny, py, py, py-1))
Mat3PosY = np.einsum("jy,kz,iyz->ijk", Dpy, Dpy, getH(Ny, py, py-1, py-1))



Bmat = np.zeros((nx, ny), dtype="float64")
AUstarmat = np.tensordot(Hpxpx, Hpypy, axes=0)


for k in range(1, nt):
    tk = k*dt

    Bmat[:, :] = np.einsum("iac,jbd,ab,cd->ij", Mat1PosX, Hpypypy, U[k-1], U[k-1])
    Bmat[:, :] += np.einsum("iac,jbd,ab,cd->ij", Mat2PosX, Mat2PosY, V[k-1], U[k-1])
    Bmat[:, :] += np.einsum("iac,jbd,ab,cd->ij", Hpxpxpx, Mat3PosY, V[k-1], V[k-1])
    # Bmat
    
    print("Bmat = ")
    print(Bmat)

    # solution = solve_system(AUstarmat, Bmat, UstarBC)
    break







k = 0




fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# Aqui calculamos o divergente de vec(u)
# divvecu = Nx[:, px-1](xplot).T @ Dpx.T @ U[k] @ Ly
# divvecu += Lx.T @ V[k] @ Dpy @ Ny[:, py-1](yplot)
# plot_field(xplot, yplot, divvecu, contour=True, ax=axes[0])
# axes[0].set_title("Divergente de vec(u)")

plt.show()