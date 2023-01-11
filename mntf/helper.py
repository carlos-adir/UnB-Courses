from compmec import nurbs
import math
import numpy as np
from typing import Callable, Tuple
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import default_rng



def getMatrix(y: int, z: int, k: int, N: nurbs.SplineBaseFunction):
    """
    Dado o par (y, z) essa funcao retorna o valor de [M_yz]
    Em que 
        (u_{k+1}-u_{k}) * [M_yz] = int_{u_k}^{u_{k+1}} [N_y] x [N_z] du
    """
    if z < y:
        return np.transpose(getMatrix(z, y, k, N))
    if y == 0 and z == 0:
        return np.ones((1, 1))
    if y == 0 and z == 1:
        return np.ones((1, 2))/2
    if y == 1 and z == 1:
        return (1+np.eye(2))/6

    U = N.knotvector
    matResult = np.zeros((y+1, z+1), dtype="float64")
    BezSysLin = np.zeros((y+z+1, y+z+1), dtype="float64")
    for ii in range(y+z+1):
        tii = ii/(y+z)
        for q in range(y+z+1):
            BezSysLin[ii, q] = math.comb(y+z,q)*(1-tii)**(y+z-q)*(tii**q)
    invBezSysLin = np.linalg.inv(BezSysLin)
    weightvector = np.array([sum(invBezSysLin[:, j]) for j in range(y+z+1)])/(y+z+1)
    
    uzvals = np.linspace(U[k], U[k+1], y+z+1)
    Nvalsy = N[k-y:k+1,y](uzvals)
    Nvalsz = N[k-z:k+1,z](uzvals)
    for ii in range(y+z+1):
        matResult += weightvector[ii] * np.tensordot(Nvalsy[:,ii], Nvalsz[:,ii], axes=0)
    return matResult

def getH(y: int, z:int, N: nurbs.SplineBaseFunction):
    U = N.knotvector
    p, n = N.degree, N.npts
    Hyz = np.zeros((n, n), dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Myz = getMatrix(y, z, k, N)
        Hyz[k-y:k+1, k-z:k+1] += (U[k+1]-U[k])*Myz
    return Hyz

def getD(j: int, U: Tuple[float]):
    n = U.npts
    alpha = np.zeros(n, dtype="float64")
    for i in range(n):
        if U[i+j] != U[i]:
            alpha[i] = j/(U[i+j]-U[i])
    Dj = np.diag(alpha)
    for i in range(n-1):
        Dj[i,i+1] = -alpha[i+1]
    return Dj



def RK4(f: Callable[[float, float], float], ti: float, h: float, wi: float):
    """Runge Kutta of order 3. Has error O(h^4)"""
    k1 = h * f(ti, wi)
    k2 = h * f(ti + h / 2, wi + k1 / 2)
    k3 = h * f(ti + h / 2, wi + k2 / 2)
    k4 = h * f(ti + h, wi + k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rungekutta(f: Callable[[float, float], float], y0: float, t: Tuple[float]):
    """
    Resolve a EDO (y' = f(t, y)) usando o método de Runge-Kutta.
    Input:
        f: Function - A função da EDO, que recebe (t, y) e retorna um float
        y0: float - A condição inicial da EDO, tal que y(t_0) = y0
        t: Array[float] - Os valores dos pontos: [t_0, t_1, ..., t_i, ..., t_n]
    Output:
        w: Array[float] - Os valores aproximados da função no ponto: [w_0, w_1, ..., w_n]
    """
    y0 = np.array(y0, dtype="object")
    t = np.array(t, dtype="object")
    n = len(t)
    w = np.zeros([n]+list(y0.shape), dtype="object")
    w_ = np.zeros([n]+list(y0.shape), dtype="object")  
    w[0] = y0
    w_[0] = f(t[0], w[0])
    for i in range(n-1):
        h = t[i+1]-t[i]
        w[i + 1] = w[i] + RK4(f, t[i], h, w[i])
        w_[i + 1] = f(t[i + 1], w[i + 1])
    return w

def plot_field(xmesh: Tuple[float], ymesh: Tuple[float], zvals: np.ndarray, ax = None):
    xmesh = np.array(xmesh, dtype="float64")
    ymesh = np.array(ymesh, dtype="float64")
    zvals = np.array(zvals, dtype="float64")
    assert xmesh.ndim == 1
    assert ymesh.ndim == 1
    assert zvals.ndim == 2
    assert zvals.shape == (len(ymesh), len(xmesh))
    x, y = np.meshgrid(xmesh, ymesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()
    dx = 0.05*(xmesh[-1]-xmesh[0])/2.
    dy = 0.05*(ymesh[-1]-ymesh[0])/2.
    extent = [xmesh[0]-dx, xmesh[-1]+dx, ymesh[0]-dy, ymesh[-1]+dy]
    im = ax.imshow(zvals[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    cp = ax.contour(x, y, zvals, 10, colors="k")
    div  = make_axes_locatable(ax)
    cax  = div.append_axes('bottom', size='5%', pad=0.6)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    return ax
    

def solve_system(A: np.ndarray, B: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve o sistema A * X = B
    Recebe 
        ```A```: Matriz de tamanho (n1, n2, n1, n2)
        ```B```: Matriz de tamanho (n1, n2)
        ```X```: Matriz de tamanho (n1, n2), com valores None dentro, com condicoes de contorno
    """
    ns = B.shape
    ndim = len(ns)
    ntot = np.prod(ns)
    mexp = (X == None).flatten()
    Aexp = np.zeros((ntot, ntot), dtype="float64")
    Bexp = np.zeros(ntot, dtype="float64")
    Xexp = np.zeros(ntot, dtype="object")
    if ndim == 1:
        Aexp[:, :] = A[:, :]
        Bexp[:] = B[:]
        Xexp[:] = X[:]
    elif ndim == 2:
        inter0 = 0
        for x in range(ns[0]):
            for y in range(ns[1]):
                Bexp[inter0] = B[x, y]
                Xexp[inter0] = X[x, y]
                inter1 = 0
                for i in range(ns[0]):
                    for j in range(ns[1]):
                        Aexp[inter0, inter1] = A[x, i, y, j]
                        inter1 += 1
                inter0 += 1
    elif ndim == 3:
        inter0 = 0
        for x in range(ns[0]):
            for y in range(ns[1]):
                for z in range(ns[2]):
                    Bexp[inter0] = B[x, y, z]
                    Xexp[inter0] = X[x, y, z]
                    inter1 = 0
                    for i in range(ns[0]):
                        for j in range(ns[1]):
                            for k in range(ns[2]):
                                Aexp[inter0, inter1] = A[x, i, y, j, z, k]
                                inter1 += 1
                    inter0 += 1
    mexp = (Xexp == None)
    Auu = np.delete(np.delete(Aexp, ~mexp, axis=0), ~mexp, axis=1)
    Aku = np.delete(np.delete(Aexp, mexp, axis=0), ~mexp, axis=1)
    Auk = np.delete(np.delete(Aexp, ~mexp, axis=0), mexp, axis=1)
    Akk = np.delete(np.delete(Aexp, mexp, axis=0), mexp, axis=1)
    Xexp[mexp] = np.linalg.solve(Auu, (Bexp[mexp] - Auk @ Xexp[~mexp]).astype("float64"))
    Bexp[~mexp] = Aku @ Xexp[mexp] + Akk @ Xexp[~mexp]
    X = Xexp.reshape(ns)
    B = Bexp.reshape(ns)
    return X, B


if __name__ == "__main__":
    n = 5

    A = np.random.rand(n, n)
    X0 = np.random.rand(n)
    B = np.einsum("ij,j->i", A, X0)
    BCs = np.zeros(n, dtype="bool")
    allchoices = [a for a in range(n)]
    numbers = default_rng().choice(allchoices, size=np.random.randint(1, n), replace=False)
    for i in numbers:
        BCs[i] = True
    Xt = np.empty(n, dtype="object")
    Xt[BCs] = X0[BCs]
    X1, _ = solve_system(A, B, Xt)
    np.testing.assert_almost_equal(X0, X1)

    n1, n2 = 4, 5
    A = np.random.rand(n1, n1, n2, n2)
    X0 = np.random.rand(n1, n2)
    B = np.einsum("xiyj,ij->xy", A, X0)
    BCs = np.zeros((n1, n2), dtype="bool")
    allchoices = []
    for a in range(n1):
        for b in range(n2):
            allchoices.append((a, b))
    numbers = default_rng().choice(allchoices, size=np.random.randint(1, n1*n2), replace=False)
    for i, j in numbers:
        BCs[i, j] = True
    Xt = np.empty((n1, n2), dtype="object")
    Xt[BCs] = X0[BCs]
    X1, _ = solve_system(A, B, Xt)
    np.testing.assert_almost_equal(X0, X1)

    n1, n2, n3 = 4, 5, 7
    A = np.random.rand(n1, n1, n2, n2, n3, n3)
    X0 = np.random.rand(n1, n2, n3)
    B = np.einsum("xiyjzk,ijk->xyz", A, X0)
    BCs = np.zeros((n1, n2, n3), dtype="bool")
    allchoices = []
    for a in range(n1):
        for b in range(n2):
            for c in range(n3):
                allchoices.append((a, b, c))
    numbers = default_rng().choice(allchoices, size=np.random.randint(1, n1*n2*n3), replace=False)
    for i, j, k in numbers:
        BCs[i, j, k] = True
    Xt = np.empty((n1, n2, n3), dtype="object")
    Xt[BCs] = X0[BCs]
    X1, _ = solve_system(A, B, Xt)
    np.testing.assert_almost_equal(X0, X1)


