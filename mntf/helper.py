from compmec import nurbs
import math
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, Optional
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import default_rng
np.set_printoptions(4)

def getChebyshevnodes(npts: int, a: float, b: float):
    chebyshevnodes = np.cos(np.pi*np.arange(1, 2*npts, 2)/(2*npts))
    return 0.5*( a+b - (b-a)*chebyshevnodes)

def get_weightvector(nodes: Tuple[float]):
    """
    Find the inverse of the matrix A from the link bellow, given the knots
    https://math.stackexchange.com/questions/4614357/find-inverse-of-matrix-with-bezier-coefficients
    """
    npts = len(nodes)
    BezSysLin = np.zeros((npts, npts), dtype="float64")
    for ii, tii in enumerate(nodes):
        for q in range(npts):
            BezSysLin[ii, q] = math.comb(npts-1,q)*(1-tii)**(npts-1-q)*(tii**q)
    invBezSysLin = np.linalg.inv(BezSysLin)
    return np.array([sum(invBezSysLin[:, j]) for j in range(npts)])/npts



def getM1(N: nurbs.SplineBaseFunction, k: int, a: int) -> np.ndarray:
    """
    Dado o valor de "a" essa funcao retorna o valor de [M_a]
    Em que 
        (u_{k+1}-u_{k}) * [M_a] = int_{u_k}^{u_{k+1}} [N_a] du
    """
    if a == 0:
        return np.ones(1)
    if a == 1:
        return np.ones(2)/2
    U = N.knotvector
    uvals = getChebyshevnodes(a+1, 0, 1)
    weightvector = get_weightvector(uvals)
    matResult = np.zeros(a+1, dtype="float64")
    uvals = getChebyshevnodes(a+1, U[k], U[k+1])
    Nvalsa = N[k-a:k+1,a](uvals)
    for aa in range(a+1):
        matResult[aa] += np.sum(weightvector * Nvalsa[aa])
    return matResult

def getM2(N: nurbs.SplineBaseFunction, k: int, a: int, b: int) -> np.ndarray:
    """
    Dado o par (a, b) essa funcao retorna o valor de [M_ab]
    Em que 
        (u_{k+1}-u_{k}) * [M_ab] = int_{u_k}^{u_{k+1}} [N_a] x [N_b] du
    """
    if a == 0 and b == 0:
        return np.ones((1, 1))
    if a == 0 and b == 1:
        return np.ones((1, 2))/2
    if a == 1 and b == 0:
        return np.ones((2, 1))/2
    if a == 1 and b == 1:
        return (1+np.eye(2))/6
    U = N.knotvector
    uvals = getChebyshevnodes(a+b+1, 0, 1)
    weightvector = get_weightvector(uvals)
    uvals = getChebyshevnodes(a+b+1, U[k], U[k+1])
    Nvalsa = N[k-a:k+1,a](uvals)
    Nvalsb = N[k-b:k+1,b](uvals)
    matResult = np.zeros((a+1, b+1), dtype="float64")
    for aa in range(a+1):
        for bb in range(b+1):
            matResult[aa, bb] = np.sum(weightvector * Nvalsa[aa] * Nvalsb[bb])
    return matResult

def getM3(N: nurbs.SplineBaseFunction, k: int, a: int, b: int, c: int) -> np.ndarray:
    """
    Dado o par (a, b, c) essa funcao retorna o valor de [M_abc]
    Em que 
        (u_{k+1}-u_{k}) * [M_abc] = int_{u_k}^{u_{k+1}} [N_a] x [N_b] x [N_c] du
    """
    # if not (a <= b and b <= c):
    #     raise ValueError(f"a, b, c must be in order: {a}, {b}, {c}")
    if a == 0 and b == 0 and c == 0:
        return np.ones((1, 1, 1))

    U = N.knotvector
    uvals = getChebyshevnodes(a+b+c+1, 0, 1)
    weightvector = get_weightvector(uvals)
    uvals = getChebyshevnodes(a+b+c+1, U[k], U[k+1])
    Nvalsa = N[k-a:k+1,a](uvals)
    Nvalsb = N[k-b:k+1,b](uvals)
    Nvalsc = N[k-c:k+1,c](uvals)
    matResult = np.zeros((a+1, b+1, c+1), dtype="float64")
    for aa in range(a+1):
        for bb in range(b+1):
            for cc in range(c+1):
                matResult[aa, bb, cc] = np.sum(weightvector * Nvalsa[aa] * Nvalsb[bb] * Nvalsc[cc])
    return matResult

def getM(N: nurbs.SplineBaseFunction, k: int, *args: Tuple[int]):
    if len(args) == 1:
        return getM1(N, k, args[0])
    if len(args) == 2:
        return getM2(N, k, args[0], args[1])
    if len(args) == 3:
        return getM3(N, k, args[0], args[1], args[2])
    raise ValueError

def getH1(N: nurbs.SplineBaseFunction, a: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Ha = np.zeros(n, dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Ma = getM1(N, k, a)
        Ha[k-a:k+1] += (U[k+1]-U[k])*Ma
    return Ha

def getH2(N: nurbs.SplineBaseFunction, a: int, b: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Hab = np.zeros((n, n), dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Mab = getM2(N, k, a, b)
        Hab[k-a:k+1, k-b:k+1] += (U[k+1]-U[k])*Mab
    return Hab

def getH3(N: nurbs.SplineBaseFunction, a: int, b:int, c: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Habc = np.zeros((n, n, n), dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Mabc = getM3(N, k, a, b, c)
        Habc[k-a:k+1, k-b:k+1, k-c:k+1] += (U[k+1]-U[k])*Mabc
    return Habc


def getH(N: nurbs.SplineBaseFunction, *args: Tuple[int]):
    if len(args) == 1:
        return getH1(N, args[0])
    if len(args) == 2:
        return getH2(N, args[0], args[1])
    if len(args) == 3:
        return getH3(N, args[0], args[1], args[2])
    raise ValueError

def getMatrix(N: nurbs.SplineBaseFunction, *args: Tuple[int]):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] x ... x [Nx^(z)] dx
    em que args = (a, b, ..., z)
    Exemplos:
        getMatrix(N, 0) -> I = int_0^1 [Nx] dx
        getMatrix(N, 0, 0) -> I = int_0^1 [Nx] x [Nx] dx
        getMatrix(N, 0, 0, 0) -> I = int_0^1 [Nx] x [Nx] x [Nx] dx
        getMatrix(N, 1) -> I = int_0^1 [Nx'] dx
        getMatrix(N, 2) -> I = int_0^1 [Nx''] dx
        getMatrix(N, 1, 0) -> I = int_0^1 [Nx'] x [Nx] dx
        getMatrix(N, 0, 1) -> I = int_0^1 [Nx] x [Nx'] dx
        getMatrix(N, 1, 1) -> I = int_0^1 [Nx'] x [Nx'] dx
    """
    n, p, U = N.npts, N.degree, N.knotvector
    ndim = len(args)
    if not (0 < ndim < 4):
        raise ValueError("The number of arguments must be 1, 2, or 3")
    maxderoriginal = np.max(args)
    maxderpartes = int(np.ceil(np.mean(args)))
    if maxderpartes > p:
        return np.zeros(ndim*[n], dtype="float64")
    D = {}
    for i in range(max(1,p-maxderpartes), p+1):
        D[i] = getD(i, U)
    if ndim == 1:
        a = a
        if a == 0:
            return getH(N, p)
        elif a == 1:
            return D[p] @ getH(N, p-1)
        elif a == 2:
            return D[p] @ D[p-1] @ getH(N, p-2)
        elif a == 3:
            return D[p] @ D[p-1] @ D[p-2] @ getH(N, p-3)
    elif ndim == 2:
        a, b = args
        if a == 0 and b == 0:
            return getH(N, p, p)
        elif a == 1 and b == 1:
            return D[p] @ getH(N, p-1, p-1) @ D[p].T
        elif a == 2 and b == 2:
            return D[p] @ D[p-1] @ getH(N, p-2, p-2) @ D[p-1].T @ D[p].T
        elif a == 3 and b == 3:
            return D[p] @ D[p-1] @ D[p-1] @ getH(N, p-3, p-3) @ D[p-2].T @ D[p-1].T @ D[p].T

        elif a == 1 and b == 0:
            return D[p] @ getH(N, p-1, p)
        elif a == 0 and b == 1:
            return getH(N, p-1, p-1) @ D[p].T
        elif a == 2 and b == 0:
            termo = np.tensordot(D[p] @ N[:, p-1](1), N[:, p](1), axes=0)
            termo -= np.tensordot(D[p] @ N[:, p-1](0), N[:, p](0), axes=0)
            termo -= getMatrix(N, 1, 1)
            return termo
        elif a == 0 and b == 2:
            termo = np.tensordot(N[:, p](1), D[p] @ N[:, p-1](1), axes=0)
            termo -= np.tensordot(N[:, p](0), D[p] @ N[:, p-1](0), axes=0)
            termo -= getMatrix(N, 1, 1)
            return termo
        elif a == 1 and b == 2:
            return D[p] @ getH(N, p-1, p-2) @ D[p-1].T @ D[p].T
        elif a == 2 and b == 1:
            return D[p] @ D[p-1] @ getH(N, p-2, p-1) @ D[p].T
    elif ndim == 3:
        a, b, c = args
        if maxderpartes == maxderoriginal:
            termo = getH(N, p-a, p-b, p-c)
            for i in range(p-a, p):
                termo = np.einsum("ia,ajk->ijk", D[i], termo)
            for i in range(p-b, p):
                termo = np.einsum("ia,ajk->ijk", D[i], termo)
            for i in range(p-c, p):
                termo = np.einsum("ia,ajk->ijk", D[i], termo)

    raise ValueError(f"For ndim = {ndim}, args = {args}")

def getAlpha(j: int, U: Tuple[float]) -> np.ndarray:
    n = U.npts
    alpha = np.zeros(n, dtype="float64")
    for i in range(n):
        if U[i+j] != U[i]:
            alpha[i] = j/(U[i+j]-U[i])
    return alpha

def getD(j: int, U: Tuple[float]) -> np.ndarray:
    alpha = getAlpha(j, U)
    Dj = np.diag(alpha)
    for i in range(U.npts-1):
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

def plot_field(xmesh: Tuple[float], ymesh: Tuple[float], zvals: np.ndarray, ax = None, contour=True):
    xmesh = np.array(xmesh, dtype="float64")
    ymesh = np.array(ymesh, dtype="float64")
    zvals = np.array(zvals, dtype="float64")
    assert xmesh.ndim == 1
    assert ymesh.ndim == 1
    assert zvals.ndim == 2
    assert zvals.shape == (len(xmesh), len(ymesh))
    x, y = np.meshgrid(xmesh, ymesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()
    dx = 0.05*(xmesh[-1]-xmesh[0])/2.
    dy = 0.05*(ymesh[-1]-ymesh[0])/2.
    extent = [xmesh[0]-dx, xmesh[-1]+dx, ymesh[0]-dy, ymesh[-1]+dy]
    im = ax.imshow((zvals.T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    if contour:
        cp = ax.contour(x, y, zvals.T, 10, colors="k")
    div  = make_axes_locatable(ax)
    cax  = div.append_axes('bottom', size='5%', pad=0.6)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    return ax
    

def solve_system(A: np.ndarray, B: np.ndarray, X: Optional[np.ndarray] = None, X0: Optional[np.ndarray] = None, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dado um sistema do tipo [A]*[X] = [B]
    Queremos encontrar o valor de X
    Mas alguns valores de X sao conhecidos, chamamos de Xk
        [Xk] são valores conhecidos
        [Xu] são valores desconhecidos 
    de forma que podemos montar o sistema
        [ [Akk]  [Aku] ]   [ [Xk] ]   [ [Bk] ]
        [              ] * [      ] = [      ]
        [ [Auk]  [Auu] ]   [ [Xu] ]   [ [Bu] ]
    Resolvemos entao
        [Auu] * [Xu] = [Bu] - [Auk] * [Xk]
        [Bk] = [Akk] * [Xk] + [Aku] * [Xu]
    Recebe 
        ```A```: Matriz de tamanho (n1, n2, n1, n2)
        ```B```: Matriz de tamanho (n1, n2)
        ```X```: Matriz de tamanho (n1, n2), com valores ```nan``` dentro e com condicoes de contorno
    Se X0 for dado, eh usado um metodo iterativo
    """
    if X is None:
        X = np.empty(B.shape, dtype="float64")
        X.fill(np.nan) 
    elif B.shape != X.shape:
        raise ValueError(f"B.shape = {B.shape} != {X.shape} = X.shape")
    if X0 is not None:
        if X0.shape != X.shape:
            raise ValueError(f"X0.shape = {X0.shape} != {X.shape} = X.shape")
    if A.ndim != 2*B.ndim:
        raise ValueError(f"A.ndim = {A.ndim} != 2*{B.ndim} = 2*B.ndim")
    if np.prod(A.shape) != np.prod(B.shape)**2:
        raise ValueError(f"A.shape = {A.shape} != 2*{B.shape} = 2*B.shape")
    if mask is None:
        mask = np.isnan(X)
    if not np.any(mask):
        raise ValueError(f"At least one unknown must be given! All values of X are known")

    indexsnan = np.array(np.where(mask)).T
    indexskno = np.array(np.where(~mask)).T
    allindexs = np.array(np.where(np.ones(mask.shape, dtype="bool"))).T
    
    ns = B.shape
    ndim = len(ns)
    ntot = np.prod(ns)
    Aexp = np.zeros((ntot, ntot), dtype="float64")
    Bexp = np.zeros((ntot), dtype="float64")
    Xexp = np.zeros((ntot), dtype="float64")
    X0exp = np.zeros((ntot), dtype="float64")
    indexs = np.zeros(2*ndim, dtype="int16")
    for i, indsi in enumerate(allindexs):
        indexs[::2] = indsi
        Bexp[i] = B[tuple(indsi)]
        Xexp[i] = X[tuple(indsi)]
        if X0 is not None:
            X0exp[i] = X0[tuple(indsi)]
        for j, indsj in enumerate(allindexs):
            indexs[1::2] = indsj
            Aexp[i, j] = A[tuple(indexs)]
    mexp = mask.reshape(ntot)
    Auu = np.delete(np.delete(Aexp, ~mexp, axis=0), ~mexp, axis=1)
    Aku = np.delete(np.delete(Aexp, mexp, axis=0), ~mexp, axis=1)
    Auk = np.delete(np.delete(Aexp, ~mexp, axis=0), mexp, axis=1)
    Akk = np.delete(np.delete(Aexp, mexp, axis=0), mexp, axis=1)
    newvec = Bexp[mexp] - (Auk @ Xexp[~mexp].astype("float64"))
    if X0 is None:
        Xexp[mexp] = np.linalg.solve(Auu, newvec)  # Resolvemos diretamente
    else:
        Xexp[mexp], _ = GradienteConjugado(Auu, newvec, X0exp[mexp])
        # Xexp[mexp], _ = GaussSeidel(Auu, newvec, X0exp[mexp])

    Bexp[~mexp] = Aku @ Xexp[mexp] + Akk @ Xexp[~mexp]
    X = Xexp.reshape(ns)
    B = Bexp.reshape(ns)
    return X, B

def invert_matrix(A: np.ndarray, X: np.ndarray = None, mask: np.ndarray = None):
    """
    Dado um sistema do tipo [A]*[X] = [B]
    Queremos encontrar o valor de X
    Mas alguns valores de X sao conhecidos, chamamos de Xk
        [Xk] são valores conhecidos
        [Xu] são valores desconhecidos 
    de forma que podemos montar o sistema
        [ [Akk]  [Aku] ]   [ [Xk] ]   [ [Bk] ]
        [              ] * [      ] = [      ]
        [ [Auk]  [Auu] ]   [ [Xu] ]   [ [Bu] ]
    Resolvemos entao
        [Auu] * [Xu] = [Bu] - [Auk] * [Xk]
        [Bk] = [Akk] * [Xk] + [Aku] * [Xu]
    Em que [Bk] eh o vetor de forcas desconhecidas, mas frequentemente nao usado
    Contudo, existe todo um trabalho:
        * Cortar a matrizes [Akk], [Aku], [Auk], [Auu]
        * Cortar as matrizes [Bu]
        * Resolver o sistema [Auu] * [Xu] = [Bu] - [Auk] * [Xk]
        * Calcular [Bk] = [Akk] * [Xk] + [Aku] * [Xu]
    E que pode ser custoso se formo fazer toda vez!
    Essa funcao entao calcula a inversa de [A] que satisfaca as condicoes de contorno
        [Xu] = [Auu]^{-1} * ([Bu] - [Auk] * [Xk])
             = [Auu]^{-1} * [Bu] + (-[Auu]^{-1} * [Auk]) * [Xk]
             = [M] * [Xk] + [N] * [Bu]
        [Bk] = [Akk] * [Xk] + [Aku] * [Auu]^{-1} * ([Bu] - [Auk] * [Xk])
             = ([Akk] - [Aku] * [Auu]^{-1} * [Auk]) * [Xk] + [Aku] * [Auu]^{-1} * [Bu]
             = [F] * [Xk] + [G] * [Bu]
        # [G] = [Aku] * [Auu]^{-1}
        # [F] = [Akk] - [Aku] * [Auu]^{-1} * [Auk]
        # [M] = -[Auu]^{-1} * [Auk]
        # [N] = [Auu]^{-1}
        # [Xu] = [M] * [Xk] + [N] * [Bu]
        # [Bk] = [F] * [Xk] + [G] * [Bu]
        # [X] = [iXX] * [X] + [iXB] * [B]
        # [B] = [iBX] * [X] + [iBB] * [B]
    Logo, da pra escrever algo como
        [X] = [iXX] * [X] + [iXB] * [B]
        [B] = [iBX] * [X] + [iBB] * [B]
    Entao essa funcao retorna matrizes
        [iXX, iXB], [iBX, iBB]
    """
    ns = A.shape[::2]
    if X is None:
        X = np.empty(ns, dtype="float64")
        X.fill(np.nan)
    elif A.ndim != 2*X.ndim:
        raise ValueError(f"A.ndim = {A.ndim} != 2*{X.ndim} = 2*X.ndim")
    elif np.prod(A.shape) != np.prod(X.shape)**2:
        raise ValueError(f"A.shape = {A.shape} != 2*{X.shape} = 2*X.shape")
    if mask is None:
        mask = np.isnan(X)
    if not np.any(mask):
        raise ValueError(f"At least one unknown must be given! All values of X are known")

    indexsnan = np.array(np.where(mask)).T
    indexskno = np.array(np.where(~mask)).T
    allindexs = np.array(np.where(np.ones(mask.shape, dtype="bool"))).T
    
    ndim = len(ns)
    ntot = np.prod(ns)
    Aexp = np.zeros((ntot, ntot), dtype="float64")
    indexs = np.zeros(2*ndim, dtype="int16")
    for i, indsi in enumerate(allindexs):
        for j, indsj in enumerate(allindexs):
            indexs[::2] = indsi
            indexs[1::2] = indsj
            Aexp[i, j] = A[tuple(indexs)]
    mexp = mask.reshape(ntot)
    Auu = np.delete(np.delete(Aexp, ~mexp, axis=0), ~mexp, axis=1)
    Aku = np.delete(np.delete(Aexp, mexp, axis=0), ~mexp, axis=1)
    Auk = np.delete(np.delete(Aexp, ~mexp, axis=0), mexp, axis=1)
    Akk = np.delete(np.delete(Aexp, mexp, axis=0), mexp, axis=1)
    N = np.linalg.inv(Auu)
    M = -N @ Auk
    G = Aku @ N
    F = Akk - G @ Auk
    iXX = np.zeros(A.shape, dtype="float64")
    iXB = np.zeros(A.shape, dtype="float64")
    iBX = np.zeros(A.shape, dtype="float64")
    iBB = np.zeros(A.shape, dtype="float64")
    indexsmat = np.zeros(2*ndim, dtype="int16")
    for i, indsi in enumerate(indexskno):
        indexsmat[::2] = indsi
        indexsmat[1::2] = indsi
        iXX[tuple(indexsmat)] = 1
    for i, indsi in enumerate(indexsnan):  # find X
        indexsmat[::2] = indsi
        for j, indsj in enumerate(indexskno):  # Use X
            indexsmat[1::2] = indsj
            iXX[tuple(indexsmat)] = M[i, j]
        for j, indsj in enumerate(indexsnan):  # Use B
            indexsmat[1::2] = indsj
            iXB[tuple(indexsmat)] = N[i, j]
    # for i, indsi in enumerate(indexskno):  # find B
    #     indexsmat[::2] = indsi
    #     for j, indsj in enumerate(indexskno):  # Use X
    #         indexsmat[1::2] = indsj
    #         iBX[tuple(indexsmat)] = F[i, j]
    #     for j, indsj in enumerate(indexsnan):  # Use B
    #         indexsmat[1::2] = indsj
    #         iBB[tuple(indexsmat)] = G[i, j]
    return ((iXX, iXB), (iBX, iBB))

def GaussSeidel(A: np.ndarray, B: np.ndarray, X0: np.ndarray, atol: float = 1e-9, verbose = False) -> Tuple[np.ndarray, int]:
    n = len(B)
    iteration = 0
    itermax = 200
    Xnew = np.copy(X0)
    while True:
        if verbose:
            print(f"X[{iteration}] = ", Xnew)
        for i in range(n):
            Xnew[i] = B[i]
            Xnew[i] -= sum(A[i,:i]*Xnew[:i])
            Xnew[i] -= sum(A[i,i+1:]*X0[i+1:])
            Xnew[i] /= A[i, i]
        error = np.max(np.abs(Xnew-X0))
        if verbose:
            print("    error = %.2e" % error)
        if error < atol:
            return Xnew, iteration
        X0 = np.copy(Xnew)
        iteration += 1
        if iteration > itermax:
            error_msg = f"Gauss Seidel doesn't converge."
            raise ValueError(error_msg)

def GradienteConjugado(A: np.ndarray, B: np.ndarray, X0: np.ndarray, atol: float = 1e-9, itermax: int = 200, verbose=False) -> Tuple[np.ndarray, int]:
    atol *= len(X0)
    iteration = 0
    r0 = B - A @ X0
    residuo = np.max(np.abs(r0))
    if residuo < atol:
        return X0, 0
    rnew = np.copy(r0)
    p0 = np.copy(r0)
    Xnew = np.copy(X0)
    while True:
        if verbose:
            print(f"X[{iteration}] = ", Xnew)
        alpha = np.inner(r0, r0)
        alpha /= (p0 @ A @ p0)
        Xnew[:] = X0[:] + alpha * p0[:]
        rnew[:] = r0[:] - alpha * A @ p0
        error = np.max(np.abs(Xnew-X0))
        residuo = np.max(np.abs(rnew))
        if verbose:
            print("    error = %.2e" % error)
            print("    resid = %.2e" % residuo)
        if residuo < atol:
            return Xnew, iteration
        beta = np.inner(rnew, rnew)
        beta /= np.inner(r0, r0)
        p0 *= beta
        p0[:] += rnew[:]
        r0[:] = rnew[:]
        iteration += 1
        if iteration > itermax:
            error_msg = f"Gradiente conjugado doesn't converge."
            raise ValueError(error_msg)

def get_random_matrix_definite_positive(side: int):
    A = np.random.rand(side, side)
    A += np.transpose(A)
    eig, P = np.linalg.eig(A)
    P = np.real(P)
    eig = 2+np.random.rand(side)
    A = P.T @ np.diag(eig) @ P
    for i in range(side):
        A[i, i] = np.sum(np.abs(A[i]))+0.01
        A[i] /= 2*A[i, i]
    A += np.transpose(A)
    eigs, _ = np.linalg.eigh(A)
    assert np.all(eigs > 0)
    assert np.all(A == np.transpose(A))
    return A


class TestingAuxiliarFunctions:

    @staticmethod
    def create_random_linsys(ns: Tuple[int]):
        ndim = len(ns)
        ntot = np.prod(ns)
        Xgood = np.random.rand(*ns)
        masknan = np.zeros(ns, dtype="bool")
        numnan = np.random.randint(1, ntot)
        tempinds = np.zeros(ndim, dtype="int16")
        while np.sum(masknan) < numnan:
            for i, ni in enumerate(ns):
                tempinds[i] = np.random.randint(ni)
            masknan[tuple(tempinds)] = True
        Xboundary = np.copy(Xgood)
        Xboundary[masknan] = np.nan

        allindexs = np.array(np.where(~np.isnan(Xgood))).T

        Bsystem = np.zeros(Xgood.shape, dtype="float64")
        Aexpanded = get_random_matrix_definite_positive(ntot)
        shapeA = [item for ni in ns for item in 2*[ni]]
        Asystem = np.zeros(shapeA, dtype="float64")
        for i, indsi in enumerate(allindexs):
            value = 0
            for j, indsj in enumerate(allindexs):
                value += Aexpanded[i,j] * Xgood[tuple(indsj)]
                indexAsys = [item for sublist in zip(indsi, indsj) for item in sublist]
                Asystem[tuple(indexAsys)] = Aexpanded[i, j]
            Bsystem[tuple(indsi)] = value
        return Asystem, Bsystem, Xboundary, Xgood


def main_test_solve_direct_system():
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        for kkk in tqdm(range(ntests)):
            ns = np.array(np.random.randint(2, 4, size=ndim), dtype="int16").tolist()
            Asystem, Bsystem, Xboundary, Xgood = TestingAuxiliarFunctions.create_random_linsys(ns)
            
            Xtest = solve_system(Asystem, Bsystem, Xboundary)[0]
            np.testing.assert_almost_equal(Xtest, Xgood)

def main_test_iterative_solve_system():
    ntests = 100
    flutuation = 0.01
    for ndim in [1, 2, 3]:
        for kkk in tqdm(range(ntests)):
            ns = np.array(np.random.randint(2, 4, size=ndim), dtype="int16").tolist()
            Asystem, Bsystem, Xboundary, Xgood = TestingAuxiliarFunctions.create_random_linsys(ns)
            masknan = np.isnan(Xboundary)

            Xinit = Xgood + flutuation*(2*np.random.rand()-1)
            Xinit[~masknan] = Xgood[~masknan]
            Xtest = solve_system(Asystem, Bsystem, Xboundary, Xinit)[0]
            np.testing.assert_almost_equal(Xtest[~masknan], Xgood[~masknan])
            np.testing.assert_almost_equal(Xtest[masknan], Xgood[masknan])
            np.testing.assert_almost_equal(Xtest, Xgood)

def main_test_invert_matrix():
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        for kkk in tqdm(range(ntests)):
            ns = np.array(np.random.randint(2, 4, size=ndim), dtype="int16").tolist()
            Asystem, Bsystem, Xboundary, Xgood = TestingAuxiliarFunctions.create_random_linsys(ns)
            
            iXX, iXB = invert_matrix(Asystem, Xboundary)[0]
            Xboundary[np.isnan(Xboundary)] = 0
            if ndim == 1:
                Xtest = np.einsum("ia,a->i", iXX, Xboundary)
                Xtest += np.einsum("ia,a->i", iXB, Bsystem)
            if ndim == 2:
                Xtest = np.einsum("iajb,ab->ij", iXX, Xboundary)
                Xtest += np.einsum("iajb,ab->ij", iXB, Bsystem)
            if ndim == 3:
                Xtest = np.einsum("iajbkc,abc->ijk", iXX, Xboundary)
                Xtest += np.einsum("iajbkc,abc->ijk", iXB, Bsystem)
            if ndim == 4:
                Xtest = np.einsum("iajbkcld,abcd->ijkl", iXX, Xboundary)
                Xtest += np.einsum("iajbkcld,abcd->ijkl", iXB, Bsystem)
            if ndim == 5:
                Xtest = np.einsum("iajbkcldpe,abcde->ijklp", iXX, Xboundary)
                Xtest += np.einsum("iajbkcldpe,abcde->ijklp", iXB, Bsystem)
            np.testing.assert_almost_equal(Xtest, Xgood)

if __name__ == "__main__":
    # main_test_iterative_solve_system()
    main_test_solve_direct_system()
    main_test_invert_matrix()