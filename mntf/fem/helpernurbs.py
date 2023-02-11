from compmec import nurbs
import numpy as np
from typing import Tuple
import math

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

def getJ1(N: nurbs.SplineBaseFunction, a: int):
    n, p, U = N.npts, N.degree, N.knotvector
    if a > p:
        return np.zeros(n, dtype="float64")
    D = {}
    for i in range(1, p+1):
        D[i] = getD(i, U)
    termo = getH(N, p-a)
    for i in range(p+1-a, p+1):
        termo = D[i] @ termo
    return termo

def getJ2(N: nurbs.SplineBaseFunction, a: int, b: int):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] dx
    Exemplos:
        getJ(N, 0, 0) -> I = int_0^1 [Nx] x [Nx] dx
        getJ(N, 1, 0) -> I = int_0^1 [Nx'] x [Nx] dx
        getJ(N, 0, 1) -> I = int_0^1 [Nx] x [Nx'] dx
        getJ(N, 1, 1) -> I = int_0^1 [Nx'] x [Nx'] dx
    """
    n, p, U = N.npts, N.degree, N.knotvector
    maxderoriginal = max(a, b)
    maxderpartes = int(np.ceil(np.mean([a, b])))
    D = {}
    for i in range(max(1, min(p+1-a, p+1-b)), p+1):
        D[i] = getD(i, U)
    if maxderpartes == maxderoriginal:
        termo = getH(N, p-a, p-b)
        for i in range(p+1-a, p+1):
            termo = np.einsum("ia,aj->ij", D[i], termo)
        for i in range(p+1-b, p+1):
            termo = np.einsum("jb,ib->ij", D[i], termo)
        return termo
    
    if b < a:
        return np.transpose(getJ2(N, b, a))
    for i in range(1, min(p+1-a, p+1-b)):
        D[i] = getD(i, U)
    if a == 0 and b == 2:
        termo = np.tensordot(N[:, p](1), D[p] @ N[:, p-1](1), axes=0)
        termo -= np.tensordot(N[:, p](0), D[p] @ N[:, p-1](0), axes=0)
        termo -= getJ2(N, 1, 1)
        return termo
    elif a == 0 and b == 4:
        termo = D[p] @ D[p-1] @ getH(N, p-2, p-2) @ D[p-1].T @ D[p]
        if p > 2:
            termo += np.tensordot(N(1), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](1), axes=0)
            termo -= np.tensordot(N(0), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](0), axes=0)
        termo -= np.tensordot(D[p] @ N[:, p-1](1), D[p] @ D[p-1] @ N[:, p-2](1), axes=0)
        termo += np.tensordot(D[p] @ N[:, p-1](0), D[p] @ D[p-1] @ N[:, p-2](0), axes=0)
        return termo
    errormsg = f"Nao pude resolver: (a, b) = ({a}, {b})"
    raise ValueError(errormsg)

def getJ3(N: nurbs.SplineBaseFunction, a: int, b: int, c: int):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] x [Nx^(c)] dx
    Exemplos:
        getJ(N, 0, 0, 0) -> I = int_0^1 [Nx] x [Nx] x [Nx] dx
        getJ(N, 0, 0, 1) -> I = int_0^1 [Nx] x [Nx] x [Nx'] dx
        getJ(N, 0, 1, 0) -> I = int_0^1 [Nx] x [Nx'] x [Nx] dx
        getJ(N, 2, 0, 0) -> I = int_0^1 [Nx''] x [Nx] x [Nx] dx
    """
    n, p, U = N.npts, N.degree, N.knotvector
    maxderoriginal = max(a, b, c)
    maxderpartes = int(np.ceil(np.mean([a, b, c])))
    D = {}
    for i in range(max(1, min(p+1-a, p+1-b, p+1-c)), p+1):
        D[i] = getD(i, U)
    if maxderpartes == maxderoriginal:
        termo = getH(N, p-a, p-b, p-c)
        for i in range(p-a+1, p+1):
            termo = np.einsum("ia,ajk->ijk", D[i], termo)
        for i in range(p-b+1, p+1):
            termo = np.einsum("ia,ajk->ijk", D[i], termo)
        for i in range(p-c+1, p+1):
            termo = np.einsum("ia,ajk->ijk", D[i], termo)
        return termo
    
    for i in range(1, min(p+1-a, p+1-b, p+1-c)):
        D[i] = getD(i, U)
    a0, b0, c0 = np.sort(np.copy([a, b, c]))
    if a0 == a and b0 == b and c0 == c:  # The values are ordenated
        if a == 0 and b == 1 and c == 2:
            termo = np.einsum("i,j,k->ijk", N[:, p](1), D[p] @ N[:, p-1](1), D[p] @ N[:, p-1](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), D[p] @ N[:, p-1](0), D[p] @ N[:, p-1](0))
            termo -= getJ3(N, 1, 1, 1)
            termo -= np.einsum("jb,kc,ibc->ijk", D[p] @ D[p-1], D[p], getH(N, p, p-2, p-1))
            return termo
        if a == 0 and b == 0 and c == 3:
            termo = np.einsum("i,j,k->ijk", N[:, p](1), N[:, p](1), D[p] @ D[p-1] @ N[:, p-2](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), N[:, p](0), D[p] @ D[p-1] @ N[:, p-2](0))
            termo += np.einsum("i,j,k->ijk", D[p] @ N[:, p-1](1), N[:, p](1), D[p] @ N[:, p-1](1))
            termo -= np.einsum("i,j,k->ijk", D[p] @ N[:, p-1](0), N[:, p](0), D[p] @ N[:, p-1](0))
            termo += np.einsum("i,j,k->ijk", N[:, p](1), D[p] @ N[:, p-1](1), D[p] @ N[:, p-2](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), D[p] @ N[:, p-1](0), D[p] @ N[:, p-2](0))
            termo += getJ(N, 2, 0, 1) # Na'' * Nb * Nc'
            termo += 2*getJ(N, 1, 1, 1)  # Na' * Nb' * Nc'
            termo += getJ(N, 0, 2, 1)  # Na * Nb'' * Nc'
            return termo
        errormsg = f"Nao pude resolver: (a, b, c) = ({a}, {b}, {c})"
        raise ValueError(errormsg)
    termo = getJ3(N, a0, b0, c0)
    if a0 == a:  # (a0, b0, c0) = (a, c, b)
        np.swapaxes(termo, 1, 2)
        return termo
    elif b0 == b:  # (a0, b0, c0) = (c, b, a)
        np.swapaxes(termo, 0, 2)
        return termo
    elif c0 == c:  # (a0, b0, c0) = (b, a, c)
        np.swapaxes(termo, 0, 1)
        return termo
    else:  # any order, but we swap all of them
        termo = getJ3(N, a0, b0, c0)
        # First, we put a in its place
        if a == b0:
            np.swapaxes(termo, 0, 1)
        elif a == c0:
            np.swapaxes(termo, 0, 2)
        np.swapaxes(termo, 1, 2)
        return termo
        

def getJ(N: nurbs.SplineBaseFunction, *args: Tuple[int]):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] x ... x [Nx^(z)] dx
    em que args = (a, b, ..., z)
    Exemplos:
        getJ(N, 0) -> I = int_0^1 [Nx] dx
        getJ(N, 0, 0) -> I = int_0^1 [Nx] x [Nx] dx
        getJ(N, 0, 0, 0) -> I = int_0^1 [Nx] x [Nx] x [Nx] dx
        getJ(N, 1) -> I = int_0^1 [Nx'] dx
        getJ(N, 2) -> I = int_0^1 [Nx''] dx
        getJ(N, 1, 0) -> I = int_0^1 [Nx'] x [Nx] dx
        getJ(N, 0, 1) -> I = int_0^1 [Nx] x [Nx'] dx
        getJ(N, 1, 1) -> I = int_0^1 [Nx'] x [Nx'] dx
    """
    n, p = N.npts, N.degree
    ndim = len(args)
    if not (0 < ndim < 4):
        raise ValueError("The number of arguments must be 1, 2, or 3")
    maxderoriginal = np.max(args)
    maxderpartes = int(np.ceil(np.mean(args)))
    if maxderpartes > p:
        return np.zeros(ndim*[n], dtype="float64")
    if ndim == 1:
        return getJ1(N, args[0])
    elif ndim == 2:
        return getJ2(N, *args)
    elif ndim == 3:
        return getJ3(N, *args)
    raise ValueError(f"For ndim = {ndim}, args = {args}. Mas der = {maxderoriginal}, by parts = {maxderpartes}")

def getAlpha(j: int, U: Tuple[float]) -> np.ndarray:
    if not isinstance(j, int):
        raise TypeError(f"j in getAlpha must be integer, not {type(j)}")
    if j < 1:
        raise ValueError(f"j must be at least 1! j = {j}")
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