import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from compmec import nurbs
from helper import getD, getH, plot_field, solve_system
from typing import Tuple, Callable, Optional
from femnavierhelper import *
from tqdm import tqdm
np.set_printoptions(precision=2, suppress=True)


def print_matrix(M: np.ndarray, name: Optional[str] = None):
    assert M.ndim == 2
    if name:
        print(name + " = ")
    print(M.T[::-1])


##################################################################################
#                           CONFIGURACOES DO PROBLEMA                            #
##################################################################################

def upper_speed(x: float) -> float: 
    """
    Boundary condition of speed
    u(x, y=1) = upper_speed(x, t)
    """
    return np.sin(np.pi*x)**2

nx, ny = 13, 13
px, py = 3, 3
tmax, nt = 3, 10001
dt = tmax/(nt-1)
Re = 10
mu = 1/Re
nt = 11

Ux = nurbs.GeneratorKnotVector.uniform(degree=px, npts=nx)
Uy = nurbs.GeneratorKnotVector.uniform(degree=py, npts=ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)


##################################################################################
#                          MONTAGEM DE MATRIZES BASICAS                          #
##################################################################################

Hpxpx = getH(Nx, px, px)
Hpypy = getH(Ny, py, py)
Hpxpx1 = getH(Nx, px, px-1)
Hpypy1 = getH(Ny, py, py-1)
Dpx = getD(px, Ux)
Dpy = getD(py, Uy)
Xb = np.tensordot(Nx[:, px](1), Nx[:, px-1](0), axes=0)
Xb -= np.tensordot(Nx[:, px](0), Nx[:, px-1](0), axes=0)
Yb = np.tensordot(Ny[:, py](1), Ny[:, py-1](0), axes=0)
Yb -= np.tensordot(Ny[:, py](0), Ny[:, py-1](0), axes=0)
LapX = Xb @ Dpx.T - Dpx @ getH(Nx, px-1, px-1) @ Dpx.T
LapY = Yb @ Dpy.T - Dpy @ getH(Ny, py-1, py-1) @ Dpy.T
invHpx = np.linalg.inv(Hpxpx)
invHpy = np.linalg.inv(Hpypy)
Hpxpxpx = getH(Nx, px, px, px)
Hpypypy = getH(Ny, py, py, py)
Hpxpxpx1 = getH(Nx, px, px, px-1)
Hpypypy1 = getH(Ny, py, py, py-1)
MatX_UU = np.einsum("jy,kz,iyz->ijk", Dpx, Dpx, getH(Nx, px,px-1,px-1))
MatX_UV = np.einsum("kz,ijz->ijk", Dpx, Hpxpxpx1)
MatY_UV = np.einsum("jy,iky->ijk", Dpy, Hpypypy1)
MatY_VV = np.einsum("jy,kz,iyz->ijk", Dpy, Dpy, getH(Ny, py, py-1, py-1))
MatX_next = np.einsum("ka,ija->ijk", Dpx, Hpxpxpx1)
MatY_next = np.einsum("ka,ija->ijk", Dpy, Hpypypy1)


##################################################################################
#                    MATRIZES PARA RESOLVER SISTEMA LINEAR                       #
##################################################################################

# Pressao
B_pressure = np.zeros((nx, ny), dtype="float64")
A_pressure = np.zeros((nx, nx, ny, ny), dtype="float64")
Xb_pressure = np.empty((nx, ny), dtype="float64")
Xb_pressure.fill(np.nan)
A_pressure += np.tensordot(LapX, Hpypy, axes=0)
A_pressure += np.tensordot(Hpxpx, LapY, axes=0)
for j in range(ny):  # Condicao de contorno na parede esquerda
    B_pressure[0, j] = 0
    A_pressure[0, :, j, :].fill(0)
    A_pressure[0, :, j, j] = Dpx @ Nx[:, px-1](0)
for j in range(ny):  # Condicao de contorno na parede direita
    B_pressure[nx-1, j] = 0
    A_pressure[nx-1, :, j, :].fill(0)
    A_pressure[nx-1, :, j, j] = Dpx @ Nx[:, px-1](1)
for i in range(nx):  # Condicao de contorno na parede de baixo
    B_pressure[i, 0] = 0
    A_pressure[i, :, 0, :].fill(0)
    A_pressure[i, i, 0, :] = Dpy @ Ny[:, py-1](0)
for i in range(nx):  # Condicao de contorno na parede de cima
    B_pressure[i, ny-1] = 0
    A_pressure[i, :, ny-1, :].fill(0)
    A_pressure[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)
Xb_pressure[0, 0] = 0  # Para ter uma referencia do valor da pressao

# Velocidade horizontal
B_unext = np.zeros((nx, ny), dtype="float64")
A_unext = np.zeros((nx, nx, ny, ny), dtype="float64")
Xb_uspeed = np.empty((nx, ny), dtype="float64")
Xb_uspeed.fill(np.nan)
A_unext[:, :, :, :] = np.tensordot(Hpxpx, Hpypy, axes=0)
Xb_uspeed[0, :] = 0  # Condicao de contorno da parede da esquerda
Xb_uspeed[nx-1, :] = 0  # Condicao de contorno da parede da direita
Xb_uspeed[:, 0] = 0  # Condicao de contorno da parede de baixo
Utopctrlpts = fit_spline_curve(Nx, upper_speed, Xb_uspeed[:, ny-1])
Xb_uspeed[1:nx-1, ny-1] = Utopctrlpts  # Condicao de contorno da parede de cima

# Velocidade vertical v
B_vnext = np.zeros((nx, ny), dtype="float64")
A_vnext = np.zeros((nx, nx, ny, ny), dtype="float64")
Xb_vspeed = np.empty((nx, ny), dtype="float64")
Xb_vspeed.fill(np.nan)
A_vnext[:, :, :, :] = np.tensordot(Hpxpx, Hpypy, axes=0)  # original
# A_vnext[:, :, :, :] = np.tensordot(Hpxpx, Hpypy1 @ Dpy.T, axes=0)  # Tentativa de eq continuidade
Xb_vspeed[0, :] = 0  # Condicao de contorno da parede da esquerda
Xb_vspeed[nx-1, :] = 0  # Condicao de contorno da parede da direita
Xb_vspeed[:, 0] = 0  # Condicao de contorno da parede de baixo
Xb_vspeed[:, ny-1] = 0  # Condicao de contorno da parede de cima


##################################################################################
#                               CONDICOES INICIAIS                               #
##################################################################################

xsymb, ysymb = sp.symbols("x y", real=True)
Ssymb = -(xsymb*(1-xsymb)) * ysymb *(1-ysymb)  # Linhas de corrente inicial, analitico
Usymb = sp.diff(Ssymb, ysymb)  # Velocidade horizontal inicial, analitico
Vsymb = sp.diff(-Ssymb, xsymb)  # Velocidade vertical inicial, analitico
Uinit = sp.lambdify((xsymb, ysymb), Usymb)
Vinit = sp.lambdify((xsymb, ysymb), Vsymb)

U = np.zeros((nt, nx, ny), dtype="float64")  # Horiziontal speed
V = np.zeros((nt, nx, ny), dtype="float64")  # Vertical speed

U[0] = fit_spline_surface(Nx, Ny, Uinit, Xb_uspeed)
V[0] = fit_spline_surface(Nx, Ny, Vinit, Xb_vspeed)

##################################################################################
#                            AQUI COMECA AS ITERACOES                            #
##################################################################################

# dt = 0.001
k = 0
for k in tqdm(range(nt-1)):
    B_pressure[1:nx-1, 1:ny-1] = -np.einsum("iac,jbd,ab,cd->ij", MatX_UU, Hpypypy, U[k], U[k])[1:nx-1, 1:ny-1]
    B_pressure[1:nx-1, 1:ny-1] -= 2*np.einsum("iac,jbd,ab,cd->ij", MatX_UV, MatY_UV, U[k], V[k])[1:nx-1, 1:ny-1]
    B_pressure[1:nx-1, 1:ny-1] -= np.einsum("iac,jbd,ab,cd->ij", Hpxpxpx, MatY_VV, V[k], V[k])[1:nx-1, 1:ny-1]
    P, _ = solve_system(A_pressure, B_pressure, Xb_pressure)
    # P += 100*np.random.rand()  # Summing a constant to pressure must not change final result

    B_unext[:, :] = np.einsum("ia,jb,ab->ij", Hpxpx, Hpypy, U[k])
    B_unext[:, :] += dt*mu*np.einsum("ia,jb,ab", LapX, Hpypy, U[k])
    B_unext[:, :] += dt*mu*np.einsum("ia,jb,ab", Hpxpx, LapY, U[k])
    B_unext[:, :] -= dt*np.einsum("ia,jb,ab->ij", Hpxpx1 @ Dpx.T, Hpypy, P)
    # B_unext[:, :] -= dt*np.einsum("iac,jbd,ab,cd->ij", MatX_next, Hpypypy, U[k], U[k])
    # B_unext[:, :] -= dt*np.einsum("iac,jbd,ab,cd->ij", Hpxpxpx, MatY_next, V[k], U[k])
    U[k+1], _ = solve_system(A_unext, B_unext, Xb_uspeed)

    # Usando essa parte, nao satisfaz a equacao do balanco de massa
    B_vnext[:, :] = np.einsum("ia,jb,ab->ij", Hpxpx, Hpypy, V[k])
    B_vnext[:, :] += dt*mu*np.einsum("ia,jb,ab", LapX, Hpypy, V[k])
    B_vnext[:, :] += dt*mu*np.einsum("ia,jb,ab", Hpxpx, LapY, V[k])
    B_vnext[:, :] -= dt*np.einsum("ia,jb,ab->ij", Hpxpx, Hpypy1 @ Dpy.T, P)
    # B_vnext[:, :] -= dt*np.einsum("iac,jbd,ab,cd->ij", MatX_next, Hpypypy, U[k], V[k])
    # B_vnext[:, :] -= dt*np.einsum("iac,jbd,ab,cd->ij", Hpxpxpx, MatY_next, V[k], V[k])
    # Esse aqui é tentando usar a equacao da continuidade, que dá errado
    # B_vnext[:, :] = -np.einsum("ia,jb,ab->ij", Hpxpx1 @ Dpx.T, Hpypy, U[k+1, :, :])
    V[k+1], _ = solve_system(A_vnext, B_vnext, Xb_vspeed)
k += 1


k = 2

Sbound = np.empty((nx, ny), dtype="float64")
Sbound.fill(np.nan)
Sbound[0, :] = 0
Sbound[nx-1, :] = 0
Sbound[:, 0] = 0
Sbound[:, ny-1] = 0
A_corrente = np.tensordot(LapX, Hpypy, axes=0)
A_corrente += np.tensordot(Hpxpx, LapY, axes=0)
B_corrente = np.einsum("ia,jb,ab->ij", Hpxpx1 @ Dpx.T, Hpypy, V[k])  # dv/dx
B_corrente = -np.einsum("ia,jb,ab->ij", Hpxpx, Hpypy1 @ Dpy.T, U[k])  # du/dy
for i in range(1, nx-1):
    A_corrente[i, :, ny-1, :].fill(0)
    A_corrente[i, i, ny-1, :] = Dpy @ Ny[:, py-1](1)
    B_corrente[i, ny-1] = Utopctrlpts[i-1]
S, _ = solve_system(A_corrente, B_corrente, Sbound)
W = invHpx @ B_corrente @ invHpy
print("S.shape = ", S.shape)
print("U.shape = ", U[k].shape)
print("V.shape = ", V[k].shape)
print("W.shape = ", W.shape)
print("P.shape = ", P.shape)
plot_all_fields(Nx, Ny, U = U[k], V=V[k], S=S, P=P, W=W)

xplot = np.linspace(0, 1, 129)
yplot = np.linspace(0, 1, 129)
divvecu = (Dpx @ Nx[:, px-1](xplot)).T @ U[k] @ Ny[:, py](yplot)
divvecu += Nx[:, px](xplot).T @ U[k] @ (Dpy @ Ny[:, py-1](yplot))
fig = plt.figure()
axes = [plt.gca()]
plot_field(xplot, yplot, divvecu, contour=True, ax=axes[0])
# axes[0].set_title("Divergente de vec(u)")

plt.show()