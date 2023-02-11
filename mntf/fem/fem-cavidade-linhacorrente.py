import numpy as np
from compmec import nurbs
from helper import getAlpha, getD, getH, getJ, solve_system, invert_matrix
from femnavierhelper import Fit, plot_all_fields, plot_field
from typing import Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from save_to_paraview import SaveParaview
np.set_printoptions(precision=3, suppress=True)

def print_matrix(M: np.ndarray, name: Optional[str] = None):
    assert M.ndim == 2
    if name:
        print(name + " = ")
    print(M.T[::-1])

def top_speed(x: float) -> float:
    return np.sin(np.pi*x)**2


###########################################
#                  MALHA                  #
###########################################

px, py = 3, 3
nx, ny = 7, 7
tmax, dtmax = 3, 0.001
ntsave = 51
dtmax = min(dtmax, tmax/(ntsave-1))
timesave = np.linspace(0, tmax, ntsave)
nt = int(np.ceil(tmax/dtmax))
dt = tmax/(nt-1)

# raise ValueError

Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)
# Nx.knot_insert(0.01)
# Nx.knot_insert(0.99)
# Ny.knot_insert(0.01)
# Ny.knot_insert(0.99)
# Ny.knot_insert(0.995)
# Ny.knot_insert(0.999)
Ux = Nx.knotvector
Uy = Ny.knotvector
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)
nx, ny = Nx.npts, Ny.npts

print("px, py = ", px, py)
print("nx, ny = ", nx, ny)
print("knot vector on X:")
print(Nx.knotvector)
print("Mesh on X: ", Ux.knots)
print("Mesh on Y: ", Uy.knots)
print("Total time: %.3f" % tmax)
print("    dt max = %.2e" % dtmax )
print("    nt save = %d" % ntsave)
print("    dt = %.3e" % dt)


###########################################
#          MONTAGEM DAS MATRIZES          #
###########################################

print("Montagem de matrizes")
Hpxpx = getJ(Nx, 0, 0)
Hpypy = getJ(Ny, 0, 0)
Hxd2x = getJ(Nx, 0, 2)
Hxd4x = getJ(Nx, 0, 4)
Hyd2y = getJ(Ny, 0, 2)
Hyd4y = getJ(Ny, 0, 4)

Mat1X = getJ(Nx, 0, 1, 0)
Mat2X = getJ(Nx, 0, 0, 1)
Mat3X = getJ(Nx, 0, 1, 2)
Mat4X = getJ(Nx, 0, 0, 3)
Mat1Y = getJ(Ny, 0, 0, 3)
Mat2Y = getJ(Ny, 0, 1, 2)
Mat3Y = getJ(Ny, 0, 0, 1)
Mat4Y = getJ(Ny, 0, 1, 0)

###########################################
#          CONDICOES DE CONTORNO          #
###########################################

print("Condicoes de contorno")
Ubound = np.zeros((nx, ny), dtype="float64")
Ubound[1:nx-1, 1:ny].fill(np.nan)
Ubound[2, ny-1] = 0
Ubound[nx-2, ny-1] = 0
Utopctrlpoints = Fit.spline_curve(Nx, top_speed, Ubound[:, ny-1])
Vbound = np.zeros((nx, ny), dtype="float64")
Vbound[1:nx-1, 1:ny-1].fill(np.nan)

Sbound = np.zeros((nx, ny), dtype="float64")
Sbound[2:nx-2, 2:ny-1].fill(np.nan)
Sbound[2:nx-2, ny-2] = -Utopctrlpoints[2:nx-2] / getAlpha(py, Uy)[-1]
dSdtbound = np.zeros((nx, ny), dtype="float64")
dSdtbound[2:nx-2, 2:ny-2].fill(np.nan)

Asystem = np.tensordot(Hxd2x, Hpypy, axes=0)
Asystem += np.tensordot(Hpxpx, Hyd2y, axes=0)
Bsystem = np.zeros((nx, ny), dtype="float64")

(iSS, iSB), _ = invert_matrix(Asystem, dSdtbound)
masknan = np.isnan(dSdtbound)
dSdtbound[masknan] = 0

###########################################
#           CONDICOES INICIAIS            #
###########################################

print("Condicoes iniciais")
Sinit = lambda x, y: np.sin(np.pi*x)**2 * y**2 *(1-y)
# Sinit = lambda x, y: 0
S = np.zeros((ntsave, nx, ny), dtype="float64")
S[0] = Fit.spline_surface(Nx, Ny, Sinit, Sbound)

###########################################
#                ITERACOES                #
###########################################


print("Iteracoes")
dSdt = np.zeros((nx, ny), dtype="float64")
for Re in [100]:
    mu = 1/Re
    try:
        Sloaded = np.load("Rey%d_S.npy" % Re)
        
        print("Sload.shape = ", Sloaded.shape)
        if Sloaded.shape != (ntsave, nx, ny):
            raise ValueError
        S[:] = Sloaded[:] 
    except Exception as e:
        for k in tqdm(range(1, ntsave)):
            timea, timeb = timesave[k-1], timesave[k]
            nsteps = int(np.ceil((timeb-timea)/dtmax))
            dt = (timeb-timea)/nsteps
            S[k] = S[k-1]
            for kk in tqdm(range(nsteps)):
                Bsystem[:, :] = mu*np.einsum("ia,jb,ab->ij", Hxd4x, Hpypy, S[k])
                Bsystem[:, :] += 2*mu*np.einsum("ia,jb,ab->ij", Hxd2x, Hyd2y, S[k])
                Bsystem[:, :] += mu*np.einsum("ia,jb,ab->ij", Hpxpx, Hyd4y, S[k])
                
                Bsystem[:, :] += np.einsum("iac,jbd,ab,cd->ij", Mat1X, Mat1Y, S[k], S[k])
                Bsystem[:, :] -= np.einsum("iac,jbd,ab,cd->ij", Mat2X, Mat2Y, S[k], S[k])
                Bsystem[:, :] += np.einsum("iac,jbd,ab,cd->ij", Mat3X, Mat3Y, S[k], S[k])
                Bsystem[:, :] -= np.einsum("iac,jbd,ab,cd->ij", Mat4X, Mat4Y, S[k], S[k])
                # dSdt[:, :] = solve_system(Asystem, Bsystem, dSdtbound, mask=masknan)[0]
                dSdt[:, :] = np.einsum("iajb,ab->ij", iSS, dSdtbound)
                dSdt[:, :] += np.einsum("iajb,ab->ij", iSB, Bsystem)
                S[k] += dt*dSdt
        np.save("Rey%d_S.npy" % Re, S)


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

saver = SaveParaview()
saver.xmesh = xplot
saver.ymesh = yplot
saver.tmesh = timesave
saver.filename = "Rey%d.xdmf" % Re
saver.fields["S"] = np.einsum("ai,bj,kab->kij", Lx, Ly, S)
saver.fields["U"] = np.einsum("ai,bj,kab->kij", Lx, dLy, S)
saver.fields["V"] = np.einsum("ai,bj,kab->kij", -dLx, Ly, S)
saver.fields["W"] = np.einsum("ai,bj,kab->kij", -ddLx, Ly, S) + np.einsum("ai,bj,kab->kij", -Lx, ddLy, S)
saver.save()


splot = Lx.T @ S[k] @ Ly
uplot = Lx.T @ S[k] @ dLy  # U
vplot = -dLx.T @ S[k] @ Ly  # V
wplot = -(ddLx.T @ S[k] @ Ly + Lx.T @ S[k] @ ddLy)  # W

plot_all_fields(Nx, Ny, S=S[k])

fig, axes = plt.subplots(4, 6, figsize=(16, 20))
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