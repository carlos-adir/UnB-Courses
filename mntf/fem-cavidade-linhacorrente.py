import numpy as np
from compmec import nurbs
from helper import getD, getH, solve_system
from femnavierhelper import Fit 
from typing import Optional
from matplotlib import pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def print_matrix(M: np.ndarray, name: Optional[str] = None):
    assert M.ndim == 2
    if name:
        print(name + " = ")
    print(M.T[::-1])

def top_speed(x: float) -> float:
    return np.sin(np.pi*x)**2

Re = 100
mu = 1/Re

###########################################
#                  MALHA                  #
###########################################

px, py = 3, 3
nx, ny = 5, 5
tmax, dtmax = 3, 0.001
ntsave = 101
dtmax = min(dtmax, tmax/(ntsave-1))
nt = int(np.ceil(tmax/dtmax))
dt = tmax/(nt-1)

Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)

print("Mesh on X: ", list(set(Ux)))
print("Mesh on Y: ", list(set(Uy)))
print("Total time: %.3f" % tmax)
print("    dt max = %.2e" % dtmax )
print("    nt save = %d" % ntsave)
print("    dt = %.3e" % dt)


###########################################
#          MONTAGEM DAS MATRIZES          #
###########################################

Dpx = getD(px, Ux)
Dpy = getD(py, Uy)
Dpx1 = getD(px-1, Ux)
Dpy1 = getD(py-1, Uy)
Dpx2 = getD(px-2, Ux)
Dpy2 = getD(py-2, Uy)
Hpxpx = getH(Nx, px, px)
Hpypy = getH(Nx, py, py)

Hxd2x = -Dpx @ getH(Nx, px-1, px-1) @ Dpx.T
Hxd2x += np.tensordot(Nx(1), Dpx @ Nx[:, px-1](1), axes=0)
Hxd2x -= np.tensordot(Nx(1), Dpx @ Nx[:, px-1](1), axes=0)
Hxd4x = Dpx @ Dpx1 @ getH(Nx, px-2, px-2) @ Dpx1.T @ Dpx
Hxd4x += np.tensordot(Nx(1), Dpx @ Dpx1 @ Dpx2 @ Nx[:, px-3](1), axes=0)
Hxd4x -= np.tensordot(Nx(0), Dpx @ Dpx1 @ Dpx2 @ Nx[:, px-3](0), axes=0)
Hxd4x -= np.tensordot(Dpx @ Nx[:, px-1](1), Dpx @ Dpx1 @ Nx[:, px-2](1), axes=0)
Hxd4x += np.tensordot(Dpx @ Nx[:, px-1](0), Dpx @ Dpx1 @ Nx[:, px-2](0), axes=0)

Hyd2y = -Dpy @ getH(Ny, py-1, py-1) @ Dpy.T
Hyd2y += np.tensordot(Ny(1), Dpy @ Ny[:, py-1](1), axes=0)
Hyd2y -= np.tensordot(Ny(1), Dpy @ Ny[:, py-1](1), axes=0)
Hyd4y = Dpy @ Dpy1 @ getH(Ny, py-2, py-2) @ Dpy1.T @ Dpy
Hyd4y += np.tensordot(Ny(1), Dpy @ Dpy1 @ Dpy2 @ Ny[:, py-3](1), axes=0)
Hyd4y -= np.tensordot(Ny(0), Dpy @ Dpy1 @ Dpy2 @ Ny[:, py-3](0), axes=0)
Hyd4y -= np.tensordot(Dpy @ Ny[:, py-1](1), Dpy @ Dpy1 @ Ny[:, py-2](1), axes=0)
Hyd4y += np.tensordot(Dpy @ Ny[:, py-1](0), Dpy @ Dpy1 @ Ny[:, py-2](0), axes=0)



###########################################
#          CONDICOES DE CONTORNO          #
###########################################

Sbound = np.zeros((nx, ny), dtype="float64")
Sbound[2:nx-2, 2:ny-1].fill(np.nan)
Ubound = np.zeros((nx, ny), dtype="float64")
Ubound[1:nx-1, 1:ny].fill(np.nan)
Utopctrlpoints = Fit.spline_curve(Nx, top_speed, Ubound[:, ny-1])

Asystem = np.tensordot(Hxd2x, Hpypy, axes=0)
Asystem += np.tensordot(Hpxpx, Hyd2y, axes=0)
Bsystem = np.zeros((nx, ny), dtype="float64")
for i in range(1, nx-1):
    # Upper boundary
    # dS/dy = U(x)
    Asystem[i, :, ny-1, :].fill(0)
    Asystem[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)
Bsystem[:, ny-1] = Utopctrlpoints

###########################################
#           CONDICOES INICIAIS            #
###########################################

Sinit = lambda x, y: -np.sin(np.pi*x)**2 * y**2 *(1-y)
S = np.zeros((ntsave, nx, ny), dtype="float64")
S[0] = Fit.spline_surface(Nx, Ny, Sinit, Sbound)



###########################################
#                ITERACOES                #
###########################################

k = 0
print(f"U top = ")
print(Utopctrlpoints)
print_matrix(Sbound, "S boundary")
print(f"S[{k}]")
print_matrix(S[k])
if True:
# for k in range():

    # Bsystem[:, :] = np.einsum()
    dSdt, _ = solve_system(Asystem, Bsystem, Sbound)
    print("dSdt = ")
    print_matrix(dSdt)
    Bsystem[:, :] = np.einsum("ia,jb,ab->ij", Hxd4x, Hpypy, S[0])
    Bsystem[:, :] += 2*np.einsum("ia,jb,ab->ij", Hxd2x, Hyd2y, S[0])
    Bsystem[:, :] += np.einsum("ia,jb,ab->ij", Hpxpx, Hyd4y, S[0])
    Bsystem[:, ny-1] = Utopctrlpoints  # Top boundary conditions
    
    newS = S[k] + dt*dSdt
    print("newS = ")
    print_matrix(newS)


# xplot = np.linspace(0, 1, 129)
# plt.plot(xplot, top_speed(xplot), label=r"original")
# plt.plot(xplot, Nx(xplot).T @ Utopctrlpoints, label=r"fit")
# plt.show()