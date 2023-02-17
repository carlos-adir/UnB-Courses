import numpy as np
from matplotlib import pyplot as plt
from helper import getJ, getD, solve_system, plot_field, getAlpha
from femnavierhelper import Fit
from save_to_paraview import SaveParaview
from compmec import nurbs

def file_exists(filename: str) -> bool:
    try:
        with open(filename, "r") as file:
            pass
        return True
    except Exception as e:
        return False

def mountA(Nx, Ny, Nt) -> np.ndarray:
    Hxd0x = getJ(Nx, 0, 0)
    Hxd2x = getJ(Nx, 0, 2)
    Hxd4x = getJ(Nx, 0, 4)
    Hyd0y = getJ(Ny, 0, 0)
    Hyd2y = getJ(Ny, 0, 2)
    Hyd4y = getJ(Ny, 0, 4)
    Htd0t = getJ(Nt, 0, 0)
    Htd1t = getJ(Nt, 0, 1)
    A = np.einsum("ia,jb,kc->kciajb", Hxd2x, Hyd0y, Htd1t)
    A += np.einsum("ia,jb,kc->kciajb", Hxd0x, Hyd2y, Htd1t)
    A -= mu*np.einsum("ia,jb,kc->kciajb", Hxd4x, Hyd0y, Htd0t)
    A -= mu*np.einsum("ia,jb,kc->kciajb", Hxd2x, Hyd2y, Htd0t)
    A -= mu*np.einsum("ia,jb,kc->kciajb", Hxd0x, Hyd4y, Htd0t)
    return A

def mountB(Nx, Ny, Nt, xsample, ysample, tsample, f) -> np.ndarray:
    px, py, pt = Nx.degree, Ny.degree, Nt.degree
    nx, ny, nt = Nx.npts, Ny.npts, Nt.npts
    Ux, Uy, Ut = Nx.knotvector, Ny.knotvector, Nt.knotvector

    F = np.zeros((len(tsample), len(xsample), len(ysample)))
    for l, tl in enumerate(tsample):
        for i, xi in enumerate(xsample):
            for j, yj in enumerate(ysample):
                F[l, i, j] = f(xi, yj, tl)
    if np.all(F == 0):
        return np.zeros((nt, nx, ny), dtype="float64")
    Lx = Nx(xsample)
    Ly = Ny(ysample)
    Lt = Nt(tsample)
    Kx = np.linalg.solve(Lx @ Lx.T, Lx)
    Ky = np.linalg.solve(Ly @ Ly.T, Ly)
    Kt = np.linalg.solve(Lt @ Lt.T, Lt)
    F = np.tensordot(Kx, F, axes=(1, 0))  # G[i,b,c] = sum_{a} Kx[i,a]*F[a,b,d]
    F = np.tensordot(Ky, F, axes=(1, 1))  # G[i,j,c] = sum_{b} Ky[j,b]*F[i,b,d]
    F = np.tensordot(Kt, F, axes=(1, 2))  # G[i,j,l] = sum_{d} Kt[l,d]*F[i,j,d]
    F = np.tensordot(getJ(Nt, 0, 0), F, axes=(1, 0))  # B[i,j,d] = sum_{l} Hptpt[d,l]*G[i,j,l]
    F = np.tensordot(getJ(Ny, 0, 0), F, axes=(1, 1))  # B[i,b,d] = sum_{j} Hyd0y[b,j]*G[i,j,d]
    F = np.tensordot(getJ(Nx, 0, 0), F, axes=(1, 2))  # B[a,b,d] = sum_{i} Hxd0x[a,i]*G[i,b,d]
    return F  # B


def top_speed(x: float) -> float:
    return np.sin(np.pi*x)**2

print("Criando malha")
nx, ny, nt = 25, 25, 15
px, py, pt = 4, 4, 3
Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
Ut = nurbs.GeneratorKnotVector.uniform(pt, nt)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)
Nt = nurbs.SplineBaseFunction(Ut)
xsample = np.linspace(0, 1, 4*nx)
ysample = np.linspace(0, 1, 4*ny)
tsample = np.linspace(0, 1, 4*nt)

Re = 1
mu = 1/Re

S = np.empty((nt, nx, ny), dtype="float64")
S.fill(np.nan)

Sinit = lambda x, y: np.sin(np.pi*x)**2 * y**2 *(1-y)
S[:, 0, :] = 0  # S(t, 0, y) = 0, left wall
S[:, 1, :] = 0  # dSdx(t, 0, y) = 0, left wall
S[:, nx-1, :] = 0  # S(t, 1, y) = 0, right wall
S[:, nx-2, :] = 0  # dSdx(t, 1, y) = 0, right wall
S[:, :, 0] = 0  # S(t, 0, y) = 0, bottom wall
S[:, :, 1] = 0  # dSdx(t, 0, y) = 0, bottom wall
S[:, :, ny-1] = 0  # S(t, 1, y) = 0, top wall
Ubound = np.zeros(nx, dtype="float64")
Ubound[2:ny-2] = np.nan
Utopctrlpoints = Fit.spline_curve(Nx, top_speed, Ubound)
dSdytopctrlpoints = -Utopctrlpoints / getAlpha(py, Uy)[-1]
for l in range(nt):
    S[l, :, ny-2] = dSdytopctrlpoints  # dSdy(t, 1, y) = 0, right wall
S[0, :, :] = Fit.spline_surface(Nx, Ny, Sinit, S[0])  # Initial conditions at T(0, x, y) = 0

filename = "Rey%d_S-linhacorrente-Difusao.npy" % Re
if file_exists(filename):
    print("Arquivo existe! Vamos ler")
    Sopened = np.load(filename)
    if Sopened.shape == S.shape:
        S[:] = Sopened[:]
    else:
        print("    Sopened.shape = ", Sopened.shape)
        print("    S.shape = ", S.shape)
if np.any(np.isnan(S)):
    print("Montando o sistema")
    f = lambda x, y, t: 0
    A = mountA(Nx, Ny, Nt)
    B = mountB(Nx, Ny, Nt, xsample, ysample, tsample, f)
    print("Resolvendo!")
    masknan = np.isnan(S)
    S[masknan] = 0
    S, _ = solve_system(A, B, S, mask=masknan)
    np.save(filename, S)
    print("Resolvido!")

print("Pos trait!")
ntsave = 33
xplot = np.linspace(0, 1, 33)
yplot = np.linspace(0, 1, 33)
timesave = np.linspace(0, 1, ntsave)
Lx, Ly, Lt = Nx(xplot), Ny(yplot), Nt(timesave)
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

print("Saving to Paraview")
saver = SaveParaview()
saver.xmesh = xplot
saver.ymesh = yplot
saver.tmesh = timesave
saver.filename = "Rey%d-NOnonlinear.xdmf" % Re
print("   S...")
saver.fields["S"] = np.einsum("ai,bj,ck,cab->kij", Lx, Ly, Lt, S)
print("   U...")
saver.fields["U"] = np.einsum("ai,bj,ck,cab->kij", Lx, dLy, Lt, S)
print("   V...")
saver.fields["V"] = np.einsum("ai,bj,ck,cab->kij", -dLx, Ly, Lt, S)
print("   W...")
saver.fields["W"] = np.einsum("ai,bj,ck,cab->kij", -ddLx, Ly, Lt, S) + np.einsum("ai,bj,ck,cab->kij", -Lx, ddLy, Lt, S)
print("   saving...")
saver.save()

k = ntsave-1
# k = 0
print("Plotando os resultados")

xplot = np.linspace(0, 1, 129)
yplot = np.linspace(0, 1, 129)
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



fig, axes = plt.subplots(1, 4, figsize=(16, 4))

splot = Lx.T @ S[k] @ Ly
plot_field(xplot, yplot, splot, contour=True, ax=axes[0])
uplot = Lx.T @ S[k] @ dLy  # U
plot_field(xplot, yplot, uplot, contour=True, ax=axes[1])
vplot = dLx.T @ S[k] @ dLy  # V
plot_field(xplot, yplot, vplot, contour=True, ax=axes[2])
wplot = -(ddLx.T @ S[k] @ Ly + Lx.T @ S[k] @ ddLy)  # W
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
splot = Lx.T @ S[k] @ Ly
uplot = Lx.T @ S[k] @ dLy  # U
vplot = -dLx.T @ S[k] @ Ly  # V
wplot = -(ddLx.T @ S[k] @ Ly + Lx.T @ S[k] @ ddLy)  # W
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