import numpy as np
from compmec import nurbs
from helpernurbs import getAlpha, getD, getH, getChebyshevnodes, Fit
import ploter
from typing import Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper import file_exists, print_matrix
from helperlinalg import solve_system, invert_matrix, eigenvalues
from save_to_paraview import SaveParaview
np.set_printoptions(precision=3, suppress=True)



def top_speed(x: float) -> float:
    # return x**2 * (1-x)**2
    return np.sin(np.pi*x)


###########################################
#                  MALHA                  #
###########################################

px, py = 2, 2
nx, ny = 13, 13

Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
# Ux = nurbs.KnotVector([0]*(px+1) + list(getChebyshevnodes(nx-1-px, 0, 1)) + [1]*(px+1))
# Uy = nurbs.KnotVector([0]*(py+1) + list(getChebyshevnodes(ny-1-py, 0, 1)) + [1]*(py+1))
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
px, py = Nx.degree, Ny.degree

print("Mesh on X: ")
print(f"         px = {px}")
print(f"         nx = {nx}")
print(f"      dxmax = {np.max(np.array(Ux.knots[1:])-Ux.knots[:-1])}")
print(f"      knots = {np.array(Ux.knots)}")
print("Mesh on Y: ")
print(f"         py = {py}")
print(f"         ny = {ny}")
print(f"      dymax = {np.max(np.array(Uy.knots[1:])-Uy.knots[:-1])}")
print(f"      knots = {np.array(Uy.knots)}")


###########################################
#          MONTAGEM DAS MATRIZES          #
###########################################

print("Montagem de matrizes")
Hx00 = getH(Nx, 0, 0)
Hy00 = getH(Ny, 0, 0)
Hx02 = getH(Nx, 0, 2)
Hx04 = getH(Nx, 0, 4)
Hy02 = getH(Ny, 0, 2)
Hy04 = getH(Ny, 0, 4)

Hx010 = getH(Nx, 0, 1, 0)
Hx001 = getH(Nx, 0, 0, 1)
Hx012 = getH(Nx, 0, 1, 2)
Hx003 = getH(Nx, 0, 0, 3)
Hy003 = getH(Ny, 0, 0, 3)
Hy012 = getH(Ny, 0, 1, 2)
Hy001 = getH(Ny, 0, 0, 1)
Hy010 = getH(Ny, 0, 1, 0)

Difusao = np.einsum("ia,jb->iajb", Hx04, Hy00)
Difusao += 2*np.einsum("ia,jb->iajb", Hx02, Hy02)
Difusao += np.einsum("ia,jb->iajb", Hx00, Hy04)

xplot = np.linspace(0, 1, 129)
yplot = np.linspace(0, 1, 129)
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

###########################################
#          CONDICOES DE CONTORNO          #
###########################################

print("Condicoes de contorno")
Uboundtop = np.zeros(nx, dtype="float64")
Uboundtop[2:nx-2] = np.nan
Uboundtop = Fit.spline_curve(Nx, top_speed, Uboundtop)
print("    Uboundtop = ")
print("    " + str(Uboundtop))
Vbound = np.zeros((nx, ny), dtype="float64")
Vbound[1:nx-1, 1:ny-1].fill(np.nan)

Sbound = np.zeros((nx, ny), dtype="float64")
Sbound[2:nx-2, 2:ny-1].fill(np.nan)
Sbound[2:nx-2, ny-2] = -Uboundtop[2:nx-2] / getAlpha(py, Uy)[-1]
masknan = np.isnan(Sbound)
dSbound = np.zeros((nx, ny), dtype="float64")
dSbound[masknan] = np.nan

###########################################
#           CONDICOES INICIAIS            #
###########################################

print("Condicoes iniciais")
# Sinit = lambda x, y: -y**2*(1-y)*np.sin(np.pi*x)**2
S = np.zeros((nx, ny), dtype="float64")
S[2:nx-2, 2:ny-2] = np.nan
S = solve_system(Difusao, np.zeros((nx, ny), dtype="float64"), Sbound)[0]
Stemp = np.copy(S)  # S to manipulate
Sminr = np.copy(S)  # S to stores minimal residual in a loop
Smtor = np.copy(S)  # S such stores minimal residual of all
ploter.plot_all_fields(Nx, Ny, S=S, title="Before")

###########################################
#                ITERACOES                #
###########################################

def f(Difusao: np.ndarray, S: np.ndarray):
    global mu, Re
    if Re < 1:
        F = Re*np.einsum("iac,jbd,ab,cd->ij", Hx010, Hy003, S, S)
        F += Re*np.einsum("iac,jbd,ab,cd->ij", Hx012, Hy001, S, S)
        F -= Re*np.einsum("iac,jbd,ab,cd->ij", Hx001, Hy012, S, S)
        F -= Re*np.einsum("iac,jbd,ab,cd->ij", Hx003, Hy010, S, S)
        F += np.einsum("iajb,ab->ij", Difusao, S)
    else:
        F = np.einsum("iac,jbd,ab,cd->ij", Hx010, Hy003, S, S)
        F += np.einsum("iac,jbd,ab,cd->ij", Hx012, Hy001, S, S)
        F -= np.einsum("iac,jbd,ab,cd->ij", Hx001, Hy012, S, S)
        F -= np.einsum("iac,jbd,ab,cd->ij", Hx003, Hy010, S, S)
        F += mu*np.einsum("iajb,ab->ij", Difusao, S)
    return F

def gradf(Difusao: np.ndarray, S: np.ndarray):
    global mu, Re
    if Re < 1:
        gradF = Difusao
        gradF += Re*np.einsum("iac,jbd,ab->icjd", Hx010, Hy003, S)
        gradF += Re*np.einsum("iac,jbd,ab->icjd", Hx012, Hy001, S)
        gradF -= Re*np.einsum("iac,jbd,ab->icjd", Hx001, Hy012, S)
        gradF -= Re*np.einsum("iac,jbd,ab->icjd", Hx003, Hy010, S)
        gradF += Re*np.einsum("iac,jbd,cd->iajb", Hx010, Hy003, S)
        gradF += Re*np.einsum("iac,jbd,cd->iajb", Hx012, Hy001, S)
        gradF -= Re*np.einsum("iac,jbd,cd->iajb", Hx001, Hy012, S)
        gradF -= Re*np.einsum("iac,jbd,cd->iajb", Hx003, Hy010, S)
    else:
        gradF = mu*Difusao
        gradF += np.einsum("iac,jbd,ab->icjd", Hx010, Hy003, S)
        gradF += np.einsum("iac,jbd,ab->icjd", Hx012, Hy001, S)
        gradF -= np.einsum("iac,jbd,ab->icjd", Hx001, Hy012, S)
        gradF -= np.einsum("iac,jbd,ab->icjd", Hx003, Hy010, S)
        gradF += np.einsum("iac,jbd,cd->iajb", Hx010, Hy003, S)
        gradF += np.einsum("iac,jbd,cd->iajb", Hx012, Hy001, S)
        gradF -= np.einsum("iac,jbd,cd->iajb", Hx001, Hy012, S)
        gradF -= np.einsum("iac,jbd,cd->iajb", Hx003, Hy010, S)
    return gradF

def Residuo(F: np.ndarray):
    # return np.linalg.norm(F)
    value = np.einsum("ia,jb,ij,ab", Hx00, Hy00, F, F)
    return value

MAXTOLERANCE = 1e+100
MINTOLERANCE = 1e-12
NITERMAX = 21
ReMIN = 1e-3
ReAlvo = 1000
Res = 10**np.linspace(np.log10(ReMIN), np.log10(ReAlvo), 101)
for i, Rei in enumerate(Res):
    Rei = np.log10(Rei)
    exponent = int(np.floor(Rei))-1
    Rei -= exponent
    Rei = int(np.ceil(10**Rei))
    Rei *= 10**exponent
    Res[i] = Rei

Re, mu = ReMIN, 1/ReMIN
print("S = ")
print(S)
print(np.einsum("iajb,ab->ij", Difusao, S))
print("Residual for Re = 0:")
print("    ", Residuo(np.einsum("iajb,ab->ij", Difusao, S)))
print("Residual for Re = %.2e:" % ReMIN)
print("    ", Residuo(f(Difusao, S)))

# assert False
recompute = False
for www, Re in enumerate(tqdm(Res, position=0, desc="i", leave=True, ncols=80)):
    print("    Para Reynolds = ", Re)
    # S[1:] = np.nan
    suffix = "Re%.1e-px%dpy%dnx%dny%d" % (Re, px, py, nx, ny)
    begin = "cavidade-linhacorrente-solvenewtintransiente"
    numpyfilename = "results/"+ begin + "-" + suffix + ".npz"
    mu = 1/Re
    compute = True
    if not recompute and file_exists(numpyfilename):
        print("Arquivo existe! Vamos ler")
        isequal = True  # Flag
        loadedarrays = np.load(numpyfilename)
        for name, array in [("Ux", Ux), ("Uy", Uy)]:
            array = np.array(array)
            if loadedarrays[name].shape != array.shape:
                isequal = False
                break
            if not np.allclose(loadedarrays[name], array):
                isequal = False
                break
        if not isequal:
            print("    Arquivo contem malha de tipo diferente. Recalcular!")
        else:
            S[:, :] = loadedarrays["S"]
            compute = False
    if compute or recompute:
        # saver.write_at_time(Ut[0], "U", dLx.T @ S[0] @ Ly)
        # saver.write_at_time(Ut[0], "V", -Lx.T @ S[0] @ dLy)
        # saver.write_at_time(Ut[0], "W", -ddLx.T @ S[0] @ Ly - -Lx.T @ S[0] @ ddLy )
        # print("Calculando parte da matriz")
        lastresiduo = 1e+200
        resminr = 1e+200
        resmtor = 1e+200
        niter = 0
        Stemp = np.copy(S)
        maxS = np.max(S)
        for i in range(10):
            Stemp[masknan] += maxS*(2*np.random.rand(*Stemp[masknan].shape)-1)
            F = f(Difusao, Stemp)
            residuo = Residuo(F)
            if residuo < resminr:
                Sminr[:, :] = Stemp[:, :]
                resminr = residuo
        if resminr < resmtor:
            Smtor[:, :] = Sminr[:, :]
            resmtor = resminr
        S[:, :] = Sminr[:, :]
        for niter in tqdm(range(NITERMAX), position=0, desc="i", leave=True, ncols=80):
            F = f(Difusao, S)
            residuo = Residuo(F)
            if residuo < resmtor:
                resmtor = residuo
                Smtor[:, :] = S[:, :]
            if residuo > MAXTOLERANCE:
                raise ValueError("Max tolerance arrived: Residuo(F) = %.2e > %.2e = MAXTOLERANCE" % (residuo, MAXTOLERANCE))
            if np.abs(residuo) < MINTOLERANCE:
                break
            lastresiduo = residuo
            gradF = gradf(Difusao, S)
            dSk = solve_system(gradF, F, dSbound)[0]
            S -= dSk
                # raise ValueError("Number of iterations maximum arrived")
        print("Residual minimal total = %.3e" % resmtor)
        S[:, :] = Smtor[:, :]
        np.savez(numpyfilename, Ux=Ux, Uy=Uy, S=S)

ploter.plot_all_fields(Nx, Ny, S=S, title="After")
plt.show()

assert False

splot = np.zeros((k, len(xplot), len(yplot)), dtype="float64")
for zz in range(k):
    splot[zz] = Lx.T @ S[zz] @ Ly
anim = ploter.plot_animated_field(Ut[:k], xplot, yplot, splot, contour=True, timecycle=4)


k = nt-1
print(f"Plotando os resultados no tempo {Ut[k]}")
xspot = int(np.where(min(abs(xplot-0.5)) == abs(xplot-0.5))[0])
yspot = int(np.where(min(abs(yplot-0.5)) == abs(yplot-0.5))[0])
splot = Lx.T @ S[k] @ Ly
uplot = Lx.T @ S[k] @ dLy  # U
vplot = -dLx.T @ S[k] @ Ly  # V
wplot = -(ddLx.T @ S[k] @ Ly + Lx.T @ S[k] @ ddLy)  # W

ploter.plot_all_fields(Nx, Ny, S=S[k])

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