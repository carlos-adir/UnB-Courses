import numpy as np
from matplotlib import pyplot as plt
from helper import getJ, getD, solve_system, plot_field, getAlpha
from femnavierhelper import Fit
from compmec import nurbs
np.set_printoptions(precision=2, suppress=True)

def mountA(Nx, Ny) -> np.ndarray:
    Hxd0x = getJ(Nx, 0, 0)
    Hxd2x = getJ(Nx, 0, 2)
    Hxd4x = getJ(Nx, 0, 4)
    Hyd0y = getJ(Ny, 0, 0)
    Hyd2y = getJ(Ny, 0, 2)
    Hyd4y = getJ(Ny, 0, 4)
    A = np.einsum("ia,jb->iajb", Hxd4x, Hyd0y)
    A += 2*np.einsum("ia,jb->iajb", Hxd2x, Hyd2y)
    A += np.einsum("ia,jb->iajb", Hxd0x, Hyd4y)
    return A

def top_speed(x: float) -> float:
    return np.sin(np.pi*x)**2

print("Criando malha")
nx, ny = 101, 101
px, py = 3, 3
Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)

S = np.empty((nx, ny), dtype="float64")
S.fill(np.nan)

S[0, :] = 0  # S(t, 0, y) = 0, left wall
S[1, :] = 0  # dSdx(t, 0, y) = 0, left wall
S[nx-1, :] = 0  # S(t, 1, y) = 0, right wall
S[nx-2, :] = 0  # dSdx(t, 1, y) = 0, right wall
S[:, 0] = 0  # S(t, 0, y) = 0, bottom wall
S[:, 1] = 0  # dSdx(t, 0, y) = 0, bottom wall
S[:, ny-1] = 0  # S(t, 1, y) = 0, top wall
Ubound = np.zeros(nx, dtype="float64")
Ubound[2:ny-2] = np.nan
Utopctrlpoints = Fit.spline_curve(Nx, top_speed, Ubound)
dSdytopctrlpoints = -Utopctrlpoints / getAlpha(py, Uy)[-1]
S[:, ny-2] = dSdytopctrlpoints  # dSdy(t, 1, y) = 0, right wall

print("Montando o sistema")
f = lambda x, y: 0
A = mountA(Nx, Ny)
B = Fit.spline_surface(Nx, Ny, f)
print("Resolvendo!")
S, _ = solve_system(A, B, S)

print("Pos trait!")

xplot = np.linspace(0, 1, 1025)
yplot = np.linspace(0, 1, 1025)
Lx, Ly = Nx(xplot), Ny(yplot)

femvalues = Lx.T @ S @ Ly
exavalues = np.tensordot(np.sin(np.pi*xplot), np.sinh(np.pi*yplot)/np.sinh(np.pi), axes=0)

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

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

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
for i in range(4):
    # axes[i].set_xlabel(r"Dimensao $x$")
    # axes[i].set_ylabel(r"Dimensao $y$")
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)
    # [axes[i].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Nx.knotvector))[1:-1]]
    # [axes[i].axhline(y=yj, ls="dotted", color="k") for yj in list(set(Ny.knotvector))[1:-1]]


xplot = np.linspace(0, 1, 1028*2+1)
yplot = np.linspace(0, 1, 1028*2+1)
xspot = int(np.where(min(abs(xplot-0.5)) == abs(xplot-0.5))[0])
yspot = int(np.where(min(abs(yplot-0.5)) == abs(yplot-0.5))[0])
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

fig, axes = plt.subplots(4, 6, figsize=(16, 20))
splot = Lx.T @ S @ Ly
uplot = Lx.T @ S @ dLy  # U
vplot = -dLx.T @ S @ Ly  # V
wplot = -(ddLx.T @ S @ Ly + Lx.T @ S @ ddLy)  # W
axes[0, 0].set_title(r"Left wall")
axes[0, 1].set_title(r"$x = 0.5$")
axes[0, 2].set_title(r"Right wall")
axes[0, 3].set_title(r"Bottom wall")
axes[0, 4].set_title(r"$y = 0.5$")
axes[0, 5].set_title(r"Top wall")
axes[0, 0].set_ylabel(r"$\psi$")
axes[1, 0].set_ylabel(r"$u$")
axes[2, 0].set_ylabel(r"$v$")
axes[3, 0].set_ylabel(r"$w$")
for i, zplot in enumerate([splot, uplot, vplot, wplot]):
    axes[i, 0].plot(xplot, zplot[0, :])
    axes[i, 1].plot(yplot, zplot[xspot, :])
    axes[i, 2].plot(xplot, zplot[-1, :])
    axes[i, 3].plot(yplot, zplot[:, 0])
    axes[i, 4].plot(yplot, zplot[:, yspot])
    axes[i, 5].plot(yplot, zplot[:, -1])

plt.show()