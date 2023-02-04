import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
from tqdm import tqdm
from numba import jit

np.set_printoptions(precision=2, suppress=True)

nx, ny, nt = 7, 7, 10001
xmin, xmax = 0, 1
ymin, ymax = 0, 1
tmin, tmax = 0, 1
xmesh = np.linspace(0, 1, nx)
ymesh = np.linspace(0, 1, ny)
tmesh = np.linspace(0, 1, nt)
dx, dy = 1/(nx-1), 1/(ny-1)
dt = (tmax-tmin)/(nt-1) 
Re = 1  # Reynolds

if dx < dt:
    raise ValueError("The value of dx is less than dt! dx = %.1e, dt = %.1e" % (dx, dt))
if 1/np.sqrt(Re) < dx:
    raise ValueError("The value of dx is too big! dx = %.1e, 1/sqrt(Re) = %.1e" % (dx, 1/np.sqrt(Re)))
if 0.25*Re*dx**2 < dt:
    raise ValueError("The value of dt is too big! dt = %.1e, Re*dx^2/4 = %.1e" % (dt, 0.25*Re*dx**2))


@np.vectorize
def U(x):  # Condicao no topo
    return 1

def verify_has_nan_inf(value: np.ndarray, name: str):
    if not np.isfinite(value).all():
        print(f"variavel {name}")
        print(value.T[::-1])
        raise ValueError(f"{name} constains np.nan or np.inf")

u = np.zeros((nx, ny+1), dtype="float64")
v = np.zeros((nx+1, ny), dtype="float64")
p = np.zeros((nx+1, ny+1), dtype="float64")
ustar = np.copy(u)
vstar = np.copy(v)

# dudx = np.empty((nx, ny), dtype="float64")
# dudy = np.empty((nx, ny), dtype="float64")
# dvdx = np.empty((nx, ny), dtype="float64")
# dvdy = np.empty((nx, ny), dtype="float64")
divu = np.empty((nx, ny), dtype="float64")
divv = np.empty((nx, ny), dtype="float64")
meanu = np.empty((nx, ny), dtype="float64")
meanv = np.empty((nx, ny), dtype="float64")
Rvals = np.zeros((nx-1, ny-1), dtype="float64")  # Para calcular a pressao
matrix2D = np.empty((nx-1, ny-1), dtype="float64")  # Force to iterate computing pressure

overdx2 = 1/(dx**2)
overdy2 = 1/(dy**2)
overdtdx = 1/(dt*dx)
overdtdy = 1/(dt*dy)
dtover2dx = dt/(2*dx)
dtover2dy = dt/(2*dy)

def init_diagon2D(nx, ny, overdx2, overdy2):
    diagon2D = np.zeros((nx-1, ny-1), dtype="float64")  # Mesmo que o lambda
    diagon2Dx = np.zeros((nx-1, ny-1), dtype="float64")
    diagon2Dy = np.zeros((nx-1, ny-1), dtype="float64")
    diagon2D[:-1, :] -= overdx2
    diagon2D[1:, :] -= overdx2
    diagon2D[:, :-1] -= overdy2
    diagon2D[:, 1:] -= overdy2
    diagon2D[:, :] = 1/diagon2D[:, :]
    diagon2Dx[:, :] = overdx2*diagon2D[:, :]
    diagon2Dy[:, :] = overdy2*diagon2D[:, :]
    return diagon2D, diagon2Dx, diagon2Dy

# @jit(nopython=True)
def compute_ustar(u, divu, meanv, ustar, Uupper):
    divu[1:nx-1, 0:ny-1] = -2*(overdx2+overdy2)*u[1:nx-1, 0:ny-1]
    divu[1:nx-1, 0:ny-1] += overdx2*u[0:nx-2, 0:ny-1]
    divu[1:nx-1, 0:ny-1] += overdx2*u[2:nx, 0:ny-1]
    divu[1:nx-1, 0:ny-1] += overdy2*u[1:nx-1, 1:ny]
    divu[1:nx-1, 1:ny-1] += overdy2*u[1:nx-1, 0:ny-2]
    divu[1:nx-1, 0] += overdy2*u[1:nx-1,-1]
    meanv[1:nx-1, 0:ny-1] = v[1:nx-1, 0:ny-1]
    meanv[1:nx-1, 0:ny-1] += v[1:nx-1, 1:ny]
    meanv[1:nx-1, 0:ny-1] += v[0:nx-2, 0:ny-1]
    meanv[1:nx-1, 0:ny-1] += v[0:nx-2, 1:ny]
    meanv *= (1/4)
    # Comecamos o preenchimento
    ustar[1:nx-1, 0:ny-1] = u[1:nx-1, 0:ny-1]*(1 - dtover2dx*(u[2:nx, 0:ny-1] - u[1:nx-1, 0:ny-1]))
    ustar[1:nx-1, 0:ny-1] += (dt/Re)*divu[1:nx-1, 0:ny-1]
    ustar[1:nx-1, 0:ny-1] -= dtover2dy*meanv[1:nx-1, 0:ny-1]*(u[1:nx-1, 1:ny] - u[1:nx-1, 0:ny-1])
    # Condicoes de contorno
    ustar[0, 0:ny-1].fill(0)
    ustar[nx-1, 0:ny-1].fill(0)
    ustar[0:nx, -1] = -ustar[0:nx, 0]
    ustar[0:nx, ny-1] = 2*Uupper-ustar[0:nx, ny-2]

# @jit(nopython=True)
def compute_vstar(v, divv, meanu, vstar):
    divv[0:nx-1, 1:ny-1] = -2*(overdx2+overdy2)*v[0:nx-1, 1:ny-1]
    divv[0:nx-1, 1:ny-1] += overdy2*v[0:nx-1, 2:ny]
    divv[0:nx-1, 1:ny-1] += overdy2*v[0:nx-1, 0:ny-2]
    divv[0:nx-1, 1:ny-1] += overdx2*v[1:nx, 1:ny-1]
    divv[1:nx-1, 1:ny-1] += overdx2*v[0:nx-2, 1:ny-1]
    divv[0, 1:ny-1] += overdx2*v[-1,1:ny-1]
    meanu[0:nx-1, 1:ny-1] = u[0:nx-1, 1:ny-1]
    meanu[0:nx-1, 1:ny-1] += u[0:nx-1, 0:ny-2]
    meanu[0:nx-1, 1:ny-1] += u[1:nx, 1:ny-1]
    meanu[0:nx-1, 1:ny-1] += u[1:nx, 0:ny-2]
    meanu *= (1/4)
    # Comecamos o preenchimento
    vstar[0:nx-1, 1:ny-1] = v[0:nx-1, 1:ny-1]*(1-dtover2dy*(v[0:nx-1, 2:ny] - v[0:nx-1, 1:ny-1]))
    vstar[0:nx-1, 1:ny-1] += (dt/Re)*divv[0:nx-1, 1:ny-1]
    vstar[0:nx-1, 1:ny-1] -= dtover2dx*meanu[0:nx-1, 1:ny-1]*(v[1:nx, 1:ny-1] - v[0:nx-1, 1:ny-1])
    # Condicoes de contorno
    vstar[0:nx, 0].fill(0)
    vstar[0:nx, ny-1].fill(0)
    vstar[-1, 0:ny] = -vstar[0, 0:ny]
    vstar[nx-1, 0:ny] = -vstar[nx-2, 0:ny]

# @jit(nopython=True)
def compute_matrix2D(matrix2D, diagon2D):
    for i in range(0, nx-1):
        matrix2D[i, 0:ny-1] = overdtdy*(vstar[i, 1:ny]-vstar[i, 0:ny-1])
    for j in range(0, ny-1):
        matrix2D[0:nx-1, j] += overdtdx*(ustar[1:nx, j]-ustar[0:nx-1, j])
    matrix2D *= diagon2D
    return matrix2D

# @jit(nopython=True)  
def compute_pressure(matrix2D, diagon2Dx, diagon2Dy, p):
    TOLERANCE = 1e-9
    ITERMAX = 200
    iter = 0
    while True:
        breaklooptolerance = True
        for i in range(0, nx-1):
            for j in range(0, ny-1):
                R = matrix2D[i, j]
                if i == 0:
                    R -= diagon2Dx[i, j]*(p[i+1, j] - p[i, j])
                elif i == nx-2:
                    R -= diagon2Dx[i, j]*(p[i-1, j] - p[i, j])
                else:
                    R -= diagon2Dx[i, j]*(p[i-1, j] - 2*p[i, j] + p[i+1, j])
                if j == 0:
                    R -= diagon2Dy[i, j]*(p[i, j+1] - p[i, j])
                elif j == ny-2:
                    R -= diagon2Dy[i, j]*(p[i, j-1] - p[i, j])
                else:
                    R -= diagon2Dy[i, j]*(p[i, j-1] - 2*p[i, j] + p[i, j+1])
                p[i, j] += R
                breaklooptolerance *= np.abs(R) < TOLERANCE
        if breaklooptolerance:
            break
        iter += 1
        if iter == ITERMAX:
            error = "Maximum iteration " + str(ITERMAX) + " reached to compute pressure"
            print(error)
            # break
            raise ValueError
    p[0:nx-1, -1] = p[0:nx-1, 0]
    p[0:nx-1, ny-1] = p[0:nx-1, ny-2]
    p[-1, :] = p[0, :]
    p[nx-1, :] = p[nx-2, :]
    return p

# @jit(nopython=True)
def compute_newu(ustar, p, u):
    for j in range(-1, ny):
        u[1:nx-1, j] = ustar[1:nx-1, j] - (dt/dx)*(p[1:nx-1, j]-p[0:nx-2, j])

# @jit(nopython=True)
def compute_newv(vstar, p, v):
    for i in range(-1, nx):
        v[i, 1:ny-1] = vstar[i, 1:ny-1] - (dt/dy)*(p[i, 1:ny-1]-p[i, 0:ny-2])



if __name__ == "__main__":
    diagon2D, diagon2Dx, diagon2Dy = init_diagon2D(nx, ny, overdx2, overdy2)
    u.fill(0)
    v.fill(0)
    p.fill(0)

    # Apenas para compilar e nao tomar tempo:
    compute_ustar(u, divu, meanv, ustar, np.zeros(xmesh.shape))
    compute_vstar(v, divv, meanu, vstar)
    matrix2D = compute_matrix2D(matrix2D, diagon2D)
    p = compute_pressure(matrix2D, diagon2Dx, diagon2Dy, p)
    compute_newu(ustar, p, u)
    compute_newv(vstar, p, v)

    # Agora vamos ao calculo
    Uupper = 2*U(xmesh)
    u[:,ny-1] = Uupper[:]  # Boundary condition
    try:
        for k, tk in enumerate(tqdm(tmesh)):
            compute_ustar(u, divu, meanv, ustar, Uupper)
            compute_vstar(v, divv, meanu, vstar)
            compute_matrix2D(matrix2D, diagon2D)
            p = compute_pressure(matrix2D, diagon2Dx, diagon2Dy, p)
            compute_newu(ustar, p, u)
            compute_newv(vstar, p, v)
    except KeyboardInterrupt as error:
        pass
    except Exception as error:
        print("Exited at time: %.2f/%.2f" % (tk, tmax))
        print("Error = ", error)
        raise error
        
        print("nx, ny, nt = %02d, %02d, %02d" % (nx, ny, nt))
        print("dx, dy, dt = %.2f, %.2f, %.2f" % (dx, dy, dt))
        
    finally:
        np.save(f"u-nt{nt}.npy", u)
        np.save(f"v-nt{nt}.npy", v)
        np.save(f"p-nt{nt}.npy", p)
        np.save(f"ustar-nt{nt}.npy", ustar)
        np.save(f"vstar-nt{nt}.npy", vstar)
    print(f"ustar[{k}] = ")
    print(ustar.T[::-1])
    print(f"vstar[{k}] = ")
    print(vstar.T[::-1])
    print(f"p[{k}] = ")
    print(p.T[::-1])
    print(f"u[{k}] = ")
    print(u.T[::-1])
    print(f"v[{k}] = ")
    print(v.T[::-1])
        
