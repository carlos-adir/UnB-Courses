import numpy as np
from matplotlib import pyplot as plt
from compmec import nurbs
from tqdm import tqdm
from ploter import *
from helpernurbs import getD, getH, getT, Fit
from helperlinalg import solve_system
from typing import Tuple, Callable
from femnavierhelper import *
import sympy as sp
np.set_printoptions(precision=2, suppress=True)


# Analitico
xsymb, ysymb = sp.symbols("x y", real=True)
# Ssymb = -(sp.sin(sp.pi * xsymb)*sp.sin(sp.pi*ysymb))**2  # Linhas de corrente
Ssymb = -sp.sin(sp.pi*xsymb)**2 * ysymb**2 *(1-ysymb)
Usymb = sp.diff(Ssymb, ysymb)
Vsymb = sp.diff(-Ssymb, xsymb)
Wsymb = sp.diff(Vsymb, xsymb) - sp.diff(Usymb, ysymb)
laplP = 2 * (sp.diff(Ssymb, (xsymb, 2)) * sp.diff(Ssymb, (ysymb, 2)) - sp.diff(Ssymb, xsymb, ysymb))
laplP = sp.expand(laplP)
laplP = sp.simplify(laplP)
laplP = laplP.factor(laplP)
Psymb = sp.sympify(0)
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

print("Ssymb = ", Ssymb)


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

nx, ny = 21, 21
px, py = 2, 2
tmax, nt = 1, 251
dt = tmax/(nt-1)
# dt = 0.00001
Re = 10
mu = 1/Re


Ux = nurbs.GeneratorKnotVector.uniform(degree=px, npts=nx)
Uy = nurbs.GeneratorKnotVector.uniform(degree=py, npts=ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)


Uupperctrlpts = Fit.spline_curve(Nx, lambda x: np.sin(np.pi*x)**2)

xsample = np.linspace(0, 1, 3*nx)
ysample = np.linspace(0, 1, 3*ny)
Lx = Nx(xsample)
Ly = Ny(ysample)

W = np.zeros((nt, nx, ny), dtype="float64")  # Rotacional
S = np.zeros((nt, nx, ny), dtype="float64")  # Linhas de corrente
W[0] = Fit.spline_surface(Nx, Ny, w_initial)
S[0] = Fit.spline_surface(Nx, Ny, s_initial)

Hpxpx = getT(Nx, px, px)
Hpypy = getT(Ny, py, py)
Hpx1px1 = getT(Nx, px-1, px-1)
Hpy1py1 = getT(Ny, py-1, py-1)
Dpx = getD(px, Ux)
Dpy = getD(py, Uy)
Dpx1 = getD(px-1, Ux)
Dpy1 = getD(py-1, Uy)

Xb = np.tensordot(Nx[:, px](1), Nx[:, px-1](1), axes=0)
Xb -= np.tensordot(Nx[:, px](0), Nx[:, px-1](0), axes=0)
Yb = np.tensordot(Ny[:, py](1), Ny[:, py-1](1), axes=0)
Yb -= np.tensordot(Ny[:, py](0), Ny[:, py-1](0), axes=0)

MatXLap = Xb @ Dpx.T - Dpx @ Hpx1px1 @ Dpx.T
MatYLap = Yb @ Dpy.T - Dpy @ Hpy1py1 @ Dpy.T

Mat1PosX = np.einsum("izk,zj->ijk", getT(Nx, px,px-1,px), Dpx.T)
Mat1PosY = np.einsum("ijz,zk->ijk", getT(Ny, py,py,py-1), Dpy.T)
Mat2PosX = np.einsum("ijz,zk->ijk", getT(Nx, px,px,px-1), Dpx.T)
Mat2PosY = np.einsum("izk,zj->ijk", getT(Ny, py,py-1,py), Dpy.T)

Bmat = np.zeros((2, nx, ny), dtype="float64")
Amat = np.zeros((2, 2, nx, nx, ny, ny))
# Condicoes de contorno conhecidos
Xbound = np.zeros((2, nx, ny), dtype="float64")
Xbound.fill(np.nan)

Amat[0, 0] += np.tensordot(MatXLap, Hpypy, axes=0)
Amat[0, 0] += np.tensordot(Hpxpx, MatYLap, axes=0)
Amat[0, 1] += np.tensordot(Hpxpx, Hpypy, axes=0)
Amat[1, 1] += np.tensordot(Hpxpx, Hpypy, axes=0)
# To put upper boundary condition on PSI,
for i in range(1, nx-1):  
    # partial psi / partial y = U_upper (x)
    Amat[0, 0, i, :, ny-1, :].fill(0)
    Amat[0, 0, i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)
Bmat[0, :, ny-1] = Uupperctrlpts
print("Uupperctrlpts = ")
print(Uupperctrlpts)
# Boundary conditions on W
for i in range(1,nx-1):  # On bottom
    # Wi0 + sum_j (d^2Nj/dy^2)(0) * Sij = 0
    Amat[1, 1, i, :, 0, :].fill(0)
    Amat[1, 0, i, i, 0, :] = Dpy @ Dpy1 @ Ny[:, py-2](0)
    Amat[1, 1, i, i, 0, 0] = 1
    Bmat[1, i, 0] = 0
for i in range(1,nx-1):  # On top
    # Wi,ny-1 + sum_j (d^2Nj/dy^2)(1) * Sij = 0
    Amat[1, 1, i, :, ny-1, :].fill(0)
    Amat[1, 0, i, i, ny-1, :] = Dpy @ Dpy1 @ Ny[:, py-2](1)
    Amat[1, 1, i, i, ny-1, ny-1] = 1
    Bmat[1, i, ny-1] = 0
for j in range(1, ny-1):  # On left
    Amat[1, 1, 0, :, j, :].fill(0)
    Amat[1, 0, 0, :, j, j] = Dpx @ Dpx1 @ Nx[:, px-2](0)
    Amat[1, 1, 0, 0, j, j] = 1
    Bmat[1, 0, j] = 0
for j in range(1, ny-1):  # On right
    Amat[1, 1, nx-1, :, j, :].fill(0)
    Amat[1, 0, nx-1, :, j, j] = Dpx @ Dpx1 @ Nx[:, px-2](1)
    Amat[1, 1, nx-1, nx-1, j, j] = 1
    Bmat[1, nx-1, j] = 0


# Colocamos as condicoes de contorno em PSI
Xbound[0, 0, :].fill(0)  # psi(0, y) = 0
Xbound[0, nx-1, :].fill(0)  # psi(1, y) = 0
Xbound[0, :, 0].fill(0)  # psi(x, 0) = 0
# Condicoes de contorno em W, nos cantos
Xbound[1, 0, 0] = 0  # w(0, 0) = 0
Xbound[1, nx-1, 0] = 0  # w(1, 0) = 0
Xbound[1, 0, ny-1] = 0  # w(0, 1) = 0
Xbound[1, nx-1, ny-1] = 0  # w(1, 1) = 0


k = 0
print(f"W[{k}] = ")
print(W[k].T[::-1])

print(f"S[{k}] = ")
print(S[k].T[::-1])

print("Xbound = ")
print(Xbound[0].T[::-1])
print(Xbound[1].T[::-1])
for k in tqdm(range(1, nt)):
    tk = k*dt
    Bmat[1, :, :] = mu*MatXLap @ W[k-1, :, :] @ Hpypy.T
    Bmat[1, :, :] += mu*Hpxpx @ W[k-1, :, :] @ MatYLap.T
    Bmat[1, :, :] += np.einsum("iac,jbd,ab,cd->ij", Mat1PosX, Mat1PosY, S[k-1], W[k-1])
    Bmat[1, :, :] -= np.einsum("iac,jbd,ab,cd->ij", Mat2PosX, Mat2PosY, S[k-1], W[k-1])
    Bmat[1, :, :] *= dt
    Bmat[1, :, :] += np.einsum("ia,jb,ab->ij", Hpxpx, Hpypy, W[k-1])

    solution, _ = solve_system(Amat, Bmat, Xbound)
    S[k, :, :] = solution[0, :, :]
    W[k, :, :] = solution[1, :, :]

U = compute_U_from_current_line(Nx, Ny, S[k])
V = compute_V_from_current_line(Nx, Ny, S[k])
P = compute_pressure_from_current_line(Nx, Ny, S[k])

print("Na parede de baixo:")
print(W[k, :, 0] + Dpy @ Dpy1 @ Ny[:, py-2](0) @ S[k, :, :].T)
print("Na parede de cima:")
print(W[k, :, ny-1] + Dpy @ Dpy1 @ Ny[:, py-2](1) @ S[k, :, :].T)
print("Na parede da esquerda:")
print(W[k, 0, :] + Dpx @ Dpx1 @ Nx[:, px-2](0) @ S[k, :, :])
print("Na parede da direita:")
print(W[k, nx-1, :] + Dpx @ Dpx1 @ Nx[:, px-2](1) @ S[k, :, :])

plot_all_fields(Nx, Ny, S=S[k], U=U, V=V, W=W[k], P=P)
xplot = np.linspace(0, 1, 257)
dSdy = Nx(xplot).T @ S[k] @ Dpy @ Ny[:, py-1](1)
plt.figure()
plt.plot(xplot, dSdy, label=r"$dSdy$")
plt.plot(xplot, np.sin(np.pi*xplot)**2, label=r"$U_{up}$")
plt.plot(xplot, Nx(xplot).T @ Uupperctrlpts, label=r"$Uupper$")
plt.legend()
plt.show()