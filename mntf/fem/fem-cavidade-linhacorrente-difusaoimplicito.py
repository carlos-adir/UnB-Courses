import numpy as np
from compmec import nurbs
from helpernurbs import getAlpha, getD, getH, getChebyshevnodes, Fit
# from femnavierhelper import 
import ploter
from typing import Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper import file_exists, print_matrix
from helperlinalg import solve_system, invert_matrix, eigenvalues
from save_to_paraview import SaveParaview
np.set_printoptions(precision=3, suppress=True)



def top_speed(x: float) -> float:
    return np.sin(np.pi*x)**2


###########################################
#                  MALHA                  #
###########################################

px, py = 3, 3
nx, ny = 11, 11
nt = 101  # to save time
tmax, dtmax = 1, 0.001
# If dtmax is None, choose based on eigenvalues

Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
# Ux = nurbs.KnotVector([0]*(px+1) + list(getChebyshevnodes(nx-1-px, 0, 1)) + [1]*(px+1))
# Uy = nurbs.KnotVector([0]*(py+1) + list(getChebyshevnodes(ny-1-py, 0, 1)) + [1]*(py+1))

Ut = np.linspace(0, tmax, nt)
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
print("Mesh on time:")
print("      total = %.3f" % tmax)
print("    nt save = %d" % nt)
print("    dt save = %.2e" % np.max(Ut[1:]-Ut[:-1]))


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
dSdtbound = np.zeros((nx, ny), dtype="float64")

Asystem = np.tensordot(Hx02, Hy00, axes=0)
Asystem += np.tensordot(Hx00, Hy02, axes=0)
Dsystem = np.tensordot(Hx04, Hy00, axes=0)
Dsystem += 2*np.tensordot(Hx02, Hy02, axes=0)
Dsystem += np.tensordot(Hx00, Hy04, axes=0)
newAsystem = np.zeros(Asystem.shape)
adveccao = np.zeros((nx, ny), dtype="float64")

eigens = eigenvalues(Asystem, masknan)
print("    np.max(np.abs(eigns)) =", np.max(np.abs(eigens)))
print("    np.min(np.abs(eigns)) =", np.min(np.abs(eigens)))
print("    cond(Asystem) = ", np.max(np.abs(eigens))/np.min(np.abs(eigens)))
if dtmax is not None:
    dtmax = min(dtmax, np.max(Ut[1:]-Ut[:-1]), 0.1/np.max(np.abs(eigens)))
else:
    dtmax = min(np.max(Ut[1:]-Ut[:-1]), 0.1/np.max(np.abs(eigens)))
dtmax = 10**np.floor(np.log10(dtmax))
print("    dtmax = %.1e"%dtmax)

(iSS, iSB), _ = invert_matrix(Asystem, dSdtbound, mask=masknan)
print("    np.any(np.isnan(iSS)) =", np.any(np.isnan(iSS)))
print("    np.any(np.isnan(iSB)) =", np.any(np.isnan(dSdtbound)))
print("    np.any(np.isnan(dSdtbound)) =", np.any(np.isnan(iSS)))
###########################################
#           CONDICOES INICIAIS            #
###########################################

print("Condicoes iniciais")
S = np.zeros((nt, nx, ny), dtype="float64")
S.fill(np.nan)
# Ainit = np.tensordot(Hx04, Hy00, axes=0)
# Ainit += 2*np.tensordot(Hx02, Hy02, axes=0)
# Ainit = np.tensordot(Hx00, Hy04, axes=0)
# Binit = np.zeros((nx, ny), dtype="float64")
# S[0], _ = solve_system(Ainit, np.zeros((nx, ny)), Sbound)
Sinit = lambda x, y: -y**2*(1-y)*np.sin(np.pi*x)**2
Sinit = lambda x, y: 0
S[0] = Fit.spline_surface(Nx, Ny, Sinit)
print("    np.any(np.isnan(S[0])) =", np.any(np.isnan(S[0])))

###########################################
#                ITERACOES                #
###########################################




print("Iteracoes")
Ux = np.array(Ux)
Uy = np.array(Uy)
Ut = np.array(Ut)
Santes = np.zeros((nx, ny), dtype="float64")
Satual = np.zeros((nx, ny), dtype="float64")
Snext = np.zeros((nx, ny), dtype="float64")
temp = np.zeros((nx, ny), dtype="float64")
for Re in [1000]:
    print("    Para Reynolds = ", Re)
    # S[1:] = np.nan
    suffix = "Re%.1e-px%dpy%dnx%dny%dnt%dtmax%.2fdt%.2e" % (Re, px, py, nx, ny, nt, tmax, dtmax)
    begin = "cavidade-linhacorrente-difusaoimplicito"
    numpyfilename = "results/"+ begin + "-" + suffix + ".npz"
    paraviewfilename = "results/" + begin + "-" + suffix + ".xdmf"
    mu = 1/Re
    # mu = 0
    if file_exists(numpyfilename):
        print("Arquivo existe! Vamos ler")
        isequal = True  # Flag
        loadedarrays = np.load(numpyfilename)
        for name, array in [("Ux", Ux), ("Uy", Uy), ("Ut", Ut)]:
            if loadedarrays[name].shape != array.shape:
                isequal = False
                break
            if not np.allclose(loadedarrays[name], array):
                isequal = False
                break
        if not isequal:
            print("    Arquivo contem malha de tipo diferente. Recalcular!")
        else:
            S[:] = loadedarrays["S"]
            k = S.shape[0]
    mask = np.isnan(S)
    if np.any(mask):
        # print("Salvando para o Paraview")
        saver = SaveParaview()
        saver.xmesh = xplot
        saver.ymesh = yplot
        saver.filename = paraviewfilename
        saver.open_writer()
        saver.write_at_time(Ut[0], "S", Lx.T @ S[0] @ Ly)
        # saver.write_at_time(Ut[0], "U", dLx.T @ S[0] @ Ly)
        # saver.write_at_time(Ut[0], "V", -Lx.T @ S[0] @ dLy)
        # saver.write_at_time(Ut[0], "W", -ddLx.T @ S[0] @ Ly - -Lx.T @ S[0] @ ddLy )
        print("Calculando parte da matriz")
        for k in tqdm(range(1, nt), position=0, desc="i", leave=True, ncols=80):
            if np.any(np.isnan(S[k-1])):
                raise ValueError(f"There's np.nan inside S[{k-1}]")
            timea, timeb = Ut[k-1], Ut[k]
            nsteps = int(np.ceil((timeb-timea)/dtmax))
            dt = (timeb-timea)/nsteps
            if np.any(np.isnan(S[k-1])):
                raise ValueError("After calculations, there's still np.nan inside S")
            S[k] = S[k-1]
            tmesh = np.linspace(timea, timeb, nsteps+1)
            newAsystem[:, :, :, :] = Asystem - 2*dt*mu*Dsystem[:, :, :]
            for kk in tqdm(range(nsteps), position=1, desc="j", leave=False, ncols=80):
                adveccao[:, :] = np.einsum("iac,jbd,ab,cd->ij", Hx010, Hy003, S[k], S[k])
                adveccao[:, :] += np.einsum("iac,jbd,ab,cd->ij", Hx012, Hy001, S[k], S[k])
                adveccao[:, :] -= np.einsum("iac,jbd,ab,cd->ij", Hx001, Hy012, S[k], S[k])
                adveccao[:, :] -= np.einsum("iac,jbd,ab,cd->ij", Hx003, Hy010, S[k], S[k])
                temp[:, :] = 2*dt*adveccao + np.einsum("iajb,ab->ij", Asystem, Santes)
                Snext[:, :], _ = solve_system(newAsystem, temp, Sbound)
                if np.any(np.isnan(Snext)):
                    raise ValueError(f"At time {tmesh[kk]}, S[k] has np.nan inside it")
                Santes[:] = Satual[:]
                Satual[:] = Snext[:]
            S[k] = Satual[:]
            saver.write_at_time(timeb, "S", Lx.T @ S[k] @ Ly)
            # saver.write_at_time(Ut[k], "U", dLx.T @ S[k] @ Ly)
            # saver.write_at_time(Ut[k], "V", -Lx.T @ S[k] @ dLy)
            # saver.write_at_time(Ut[k], "W", -ddLx.T @ S[k] @ Ly - -Lx.T @ S[k] @ ddLy )
        saver.close_writer()
        np.savez(numpyfilename, Ux=Ux, Uy=Uy, Ut=Ut, S=S)

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



fig, axes = plt.subplots(3, 4, figsize=(16, 20))
axes[0, 0].set_title(r"$s$")
axes[0, 1].set_title(r"$u$")
axes[0, 2].set_title(r"$v$")
axes[0, 3].set_title(r"$w$")
axes[0, 0].set_ylabel(r"Left wall")
axes[1, 0].set_ylabel(r"$x=0.5$")
axes[2, 0].set_ylabel(r"Right wall")
for i, zplot in enumerate([splot, uplot, vplot, wplot]):
    axes[0, i].plot(zplot[0, :], yplot)
    axes[1, i].plot(zplot[xspot, :], yplot)
    axes[2, i].plot(zplot[-1, :], yplot)


fig, axes = plt.subplots(3,4, figsize=(16, 20))
axes[0, 0].set_title(r"$s$")
axes[0, 1].set_title(r"$u$")
axes[0, 2].set_title(r"$v$")
axes[0, 3].set_title(r"$w$")
axes[2, 0].set_ylabel(r"Bottom wall")
axes[1, 0].set_ylabel(r"$y=0.5$")
axes[0, 0].set_ylabel(r"Top wall")
for i, zplot in enumerate([splot, uplot, vplot, wplot]):
    axes[2, i].plot(xplot, zplot[:, 0])
    axes[1, i].plot(xplot, zplot[:, yspot])
    axes[0, i].plot(xplot, zplot[:, -1])


plt.show()