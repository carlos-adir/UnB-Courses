import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from compmec import nurbs
from helpernurbs import getH, Fit
from helper import file_exists, print_matrix
from ploter import plot_field, plot_all_fields, plot_animated_field
from helperlinalg import solve_system, invert_matrix
from typing import Tuple, Callable, Optional
from femnavierhelper import *
from tqdm import tqdm
from save_to_paraview import SaveParaview
np.set_printoptions(precision=2, suppress=True)

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
px, py = 1, 1
nt = 101  # to save time
tmax, dtmax = 0.25, 0.001
Ut = np.linspace(0, tmax, nt)
Ux = nurbs.GeneratorKnotVector.uniform(degree=px, npts=nx)
Uy = nurbs.GeneratorKnotVector.uniform(degree=py, npts=ny)
Nx = nurbs.SplineBaseFunction(Ux)
Ny = nurbs.SplineBaseFunction(Uy)


##################################################################################
#                          MONTAGEM DE MATRIZES BASICAS                          #
##################################################################################

Dpx = getD(px, Ux)
Dpy = getD(py, Uy)

Hx00 = getH(Nx, 0, 0)
Hx01 = getH(Nx, 0, 1)
Hx02 = getH(Nx, 0, 2)
Hy00 = getH(Ny, 0, 0)
Hy01 = getH(Ny, 0, 1)
Hy02 = getH(Ny, 0, 2)

Hx000 = getH(Nx, 0, 0, 0)
Hx001 = getH(Nx, 0, 0, 1)
Hx010 = getH(Nx, 0, 1, 0)
Hx011 = getH(Nx, 0, 1, 1)
Hy000 = getH(Ny, 0, 0, 0)
Hy001 = getH(Ny, 0, 0, 1)
Hy010 = getH(Ny, 0, 1, 0)
Hy011 = getH(Ny, 0, 1, 1)

invHx00 = np.linalg.inv(Hx00)
invHy00 = np.linalg.inv(Hy00)

##################################################################################
#                            CONDICOES DE CONTORNO                               #
##################################################################################

Ubound = np.zeros((nx, ny), dtype="float64")
Ubound[1:nx-1, 1:ny-1] = np.nan
Ubound[:, ny-1] = Fit.spline_curve(Nx, lambda x: np.sin(np.pi*x)**2)
masknanU = np.isnan(Ubound)

Vbound = np.zeros((nx, ny), dtype="float64")
Vbound[1:nx-1, 1:ny-1] = np.nan
masknanV = np.isnan(Vbound)

Pbound = np.empty((nx, ny), dtype="float64")
Pbound.fill(np.nan)
Pbound[0, 0] = 0
Pbound[nx-1, 0] = 0
Pbound[0, ny-1] = 0
Pbound[nx-1, ny-1] = 0
masknanP = np.isnan(Pbound)

##################################################################################
#                    MATRIZES PARA RESOLVER SISTEMA LINEAR                       #
##################################################################################

# Pressure
AsysP = np.tensordot(Hx00, Hy02, axes=0)
AsysP += np.tensordot(Hx02, Hy00, axes=0)
BsysP = np.zeros((nx, ny), dtype="float64")
for i in range(nx):  # Bottom
    AsysP[i, :, 0, :] = 0
    AsysP[i, i, 0, :] = Dpy @ Ny[:, py-1](0) 
for i in range(nx):  # Top
    AsysP[i, :, ny-1, :] = 0
    AsysP[i, i, ny-1, :]  = Dpy @ Ny[:, py-1](1) 
for j in range(ny):  # Left
    AsysP[0, :, j, :] = 0
    AsysP[0, :, j, j] = Dpx @ Nx[:, px-1](0) 
for j in range(ny):  # Right
    AsysP[nx-1, :, j, :] = 0
    AsysP[nx-1, :, j, j] = Dpx @ Nx[:, px-1](1) 

# Uspeed
AsysU = np.tensordot(Hx00, Hy00, axes=0)
BsysU = np.zeros((nx, ny), dtype="float64")

# Vspeed
AsysV = np.tensordot(Hx00, Hy00, axes=0)
BsysV = np.zeros((nx, ny), dtype="float64")

iUU, iUB = invert_matrix(AsysU, Ubound, masknanU)[0]
iVV, iVB = invert_matrix(AsysV, Vbound, masknanV)[0]
# Ubound[masknanU] = 0
# Vbound[masknanV] = 0

##################################################################################
#                               CONDICOES INICIAIS                               #
##################################################################################

xsymb, ysymb = sp.symbols("x y", real=True)
Ssymb = -sp.sin(sp.pi*xsymb)**2 * ysymb**2 *(1-ysymb)  # Linhas de corrente inicial, analitico
# Ssymb = sp.sympify(0)
Usymb = sp.diff(Ssymb, ysymb)  # Velocidade horizontal inicial, analitico
Vsymb = sp.diff(-Ssymb, xsymb)  # Velocidade vertical inicial, analitico
Uinit = sp.lambdify((xsymb, ysymb), Usymb)
Vinit = sp.lambdify((xsymb, ysymb), Vsymb)

U = np.empty((nt, nx, ny), dtype="float64")  # Horiziontal speed
V = np.empty((nt, nx, ny), dtype="float64")  # Vertical speed
U.fill(np.nan)
V.fill(np.nan)

U[0] = Fit.spline_surface(Nx, Ny, Uinit, Ubound)
V[0] = Fit.spline_surface(Nx, Ny, Vinit, Vbound)

Ustar = np.zeros((nx, ny), dtype="float64")
Vstar = np.zeros((nx, ny), dtype="float64")
Ptemp = np.zeros((nx, ny), dtype="float64")
Advec = np.zeros((nx, ny), dtype="float64")
Difus = np.zeros((nx, ny), dtype="float64")

##################################################################################
#                            AQUI COMECA AS ITERACOES                            #
##################################################################################

xplot = np.linspace(0, 1, 129)
yplot = np.linspace(0, 1, 129)
Lx, Ly = Nx(xplot), Ny(yplot)
print("Comecando as iteracoes")
# dt = 0
k = 0
for Re in [10]:
    mu = 1/Re
    suffix = "Re%d-px%dpy%dnx%dny%dnt%dtmax%.2fdt%.2e" % (Re, px, py, nx, ny, nt, tmax, dtmax)
    begin = "cavidade-projecao-UVP"
    
    numpyfilename = "results/"+ begin + "-" + suffix + ".npz"
    paraviewfilename = "results/" + begin + "-" + suffix + ".xdmf"
    mu = 1/Re
    mu = 0
    if file_exists(numpyfilename):
        print("Arquivo existe! Vamos ler")
        isequal = True  # Flag
        loadedarrays = np.load(numpyfilename)
        for name, array in [("Ux", Ux), ("Uy", Uy), ("Ut", Ut)]:
            if loadedarrays[name].shape != np.array(array).shape:
                isequal = False
                break
            if not np.allclose(loadedarrays[name], np.array(array)):
                isequal = False
                break
        if not isequal:
            print("    Arquivo contem malha de tipo diferente. Recalcular!")
        else:
            U[:] = loadedarrays["U"]
            V[:] = loadedarrays["V"]
            k = U.shape[0]-1
    if np.any(np.isnan(U)) or np.any(np.isnan(V)):
        # print("Salvando para o Paraview")
        saver = SaveParaview()
        saver.xmesh = xplot
        saver.ymesh = yplot
        saver.filename = paraviewfilename
        saver.open_writer()
        saver.write_at_time(Ut[0], "U", Lx.T @ U[0] @ Ly)
        saver.write_at_time(Ut[0], "V", Lx.T @ V[0] @ Ly)
        
        newU = np.copy(U[0])
        newV = np.copy(V[0])
        oldU = np.copy(U[0])
        oldV = np.copy(V[0])
        Ubound[np.isnan(Ubound)] = 0
        Vbound[np.isnan(Vbound)] = 0
        print("Calculando parte da matriz")
        for k in tqdm(range(1, nt), position=0, desc="i", leave=True, ncols=80):
            timea, timeb = Ut[k-1], Ut[k]
            nsteps = int(np.ceil((timeb-timea)/dtmax))
            dt = (timeb-timea)/nsteps
            U[k] = U[k-1]
            V[k] = V[k-1]
            if np.any(np.isnan(U[k])):
                raise ValueError(f"Domage! Got np.nan inside U[{k}]. At time {timea:.2f}")
            if np.any(np.isnan(V[k])):
                raise ValueError(f"Domage! Got np.nan inside V[{k}]. At time {timea:.2f}")

            tmesh = np.linspace(timea, timeb, nsteps+1)
            for kk in tqdm(range(nsteps), position=1, desc="j", leave=False, ncols=80):
                Difus[:, :] = Hx02 @ oldU @ Hy00 + Hx00 @ oldU @ Hy02.T
                Advec[:, :] = np.einsum("iac,jbd,ab,cd->ij", Hx001, Hy000, oldU, oldU)
                Advec[:, :] += np.einsum("iac,jbd,ab,cd->ij", Hx000, Hy001, oldV, oldU)
                # Advec.fill(0)
                Ustar[:, :] = oldU[:, :] + dt * (invHx00 @ (mu*Difus - Advec) @ invHy00)
                if np.any(np.isnan(Ustar)):
                    raise ValueError(f"Got np.nan inside Ustar[{k}]. At time {tmesh[kk]:.2f}")

                Difus[:, :] = Hx02 @ oldV @ Hy00 + Hx00 @ oldV @ Hy02.T
                Advec[:, :] = np.einsum("iac,jbd,ab,cd->ij", Hx001, Hy000, oldU, oldV)
                Advec[:, :] += np.einsum("iac,jbd,ab,cd->ij", Hx000, Hy001, oldV, oldV)
                # Advec.fill(0)
                Vstar[:, :] = oldV[:, :] + dt * (invHx00 @ (mu*Difus - Advec) @ invHy00)
                if np.any(np.isnan(Vstar)):
                    raise ValueError(f"Got np.nan inside Vstar[{k}]. At time {tmesh[kk]:.2f}")
                
                BsysP[:, :] = (Hx01 @ Ustar @ Hy00)
                BsysP[:, :] += (Hx00 @ Vstar @ Hy01.T)
                if np.any(np.isnan(BsysP)):
                    raise ValueError(f"Got np.nan inside BsysP[{k}]. At time {tmesh[kk]:.2f}")
                Ptemp[:, :] = solve_system(AsysP, BsysP, Pbound)[0]
                if np.any(np.isnan(Ptemp)):
                    raise ValueError(f"Got np.nan inside P[{k}]")

                BsysU[:, :] = Hx00 @ Ustar @ Hy00.T
                BsysU[:, :] -= Hx01 @ Ptemp @ Hy00.T
                if np.any(np.isnan(BsysU)):
                    raise ValueError(f"Got np.nan inside BsysU[{k}]. At time {tmesh[kk]:.2f}")
                # newU[:, :] = solve_system(AsysU, BsysU, Ubound, mask=masknanU)[0]
                newU = np.einsum("iajb,ab->ij", iUU, Ubound) + np.einsum("iajb,ab->ij", iUB, BsysU)
                # newU[:, :] = invHx00 @ BsysU @ invHy00
                # newU[~masknanU] = Ubound[~masknanU]
                if np.any(np.isnan(newU)):
                    raise ValueError(f"Got np.nan inside U[{k}]. At time {tmesh[kk]:.2f}")

                BsysV[:, :] = Hx00 @ Vstar @ Hy00.T
                BsysV[:, :] -= Hx00 @ Ptemp @ Hy01.T
                if np.any(np.isnan(BsysV)):
                    raise ValueError(f"Got np.nan inside BsysV[{k}]. At time {tmesh[kk]:.2f}")
                # newV = np.einsum("iajb,ab->ij", iVV, Vbound) + np.einsum("iajb,ab->ij", iVB, BsysV)
                newV[:, :] = solve_system(AsysV, BsysV, Vbound, mask=masknanV)[0]
                # newV[:, :] = invHx00 @ BsysV @ invHy00
                # newV[~masknanV] = Vbound[~masknanV]
                if np.any(np.isnan(newV)):
                    raise ValueError(f"Got np.nan inside V[{k}]. At time {tmesh[kk]:.2f}")
                
                oldU[:, :] = newU[:, :]
                oldV[:, :] = newV[:, :]
            U[k, :, :] = newU[:, :]
            V[k, :, :] = newV[:, :]
            saver.write_at_time(timeb, "U", Lx.T @ U[k] @ Ly)
            saver.write_at_time(timeb, "V", Lx.T @ V[k] @ Ly)
        saver.close_writer()
        np.savez(numpyfilename, Ux=np.array(Ux), Uy=np.array(Uy), Ut=Ut, U=U, V=V)

print(f"U[{k}]")
print(U[k].T[::-1])
print(f"V[{k}]")
print(V[k].T[::-1])

Sbound = np.zeros((nx, ny), dtype="float64")
Sbound[2:nx-2, 2:ny-2] = np.nan
BsysS = Hx00 @ U[k] @ Hy01.T - Hx01 @ V[k] @ Hy00.T
AsysS = np.tensordot(Hx00, Hy00, axes=0)
S = solve_system(AsysS, BsysS, Sbound)[0]
print("S = ")
print(S)


alluvals = np.zeros((nt, len(xplot), len(yplot)), dtype="float64")
allvvals = np.zeros((nt, len(xplot), len(yplot)), dtype="float64")
for k, tk in enumerate(Ut):
    alluvals[k] = Lx.T @ U[k] @ Ly
    allvvals[k] = Lx.T @ V[k] @ Ly
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# plot_field(xmesh, ymesh, uvals, ax=axes[0], contour=True)
# plot_field(xmesh, ymesh, vvals, ax=axes[1], contour=True)
anim = plot_animated_field(Ut, xplot, yplot, alluvals, contour=True)
plt.show()
assert False
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