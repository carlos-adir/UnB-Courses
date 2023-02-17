from compmec import nurbs
import numpy as np
from typing import Tuple, Callable
import math
from matplotlib import pyplot as plt

class Fit:

    @staticmethod
    def spline_curve(N: nurbs.SplineBaseFunction, f: Callable[[float], float], BCvals: np.ndarray=None) -> np.array:
        tsample = []
        knots = N.knotvector.knots
        for ta, tb in zip(knots[:-1], knots[1:]):
            tsample.extend(np.linspace(ta, tb, 5, endpoint=False))
        tsample.append(1)
        tsample = np.array(tsample)
        Lt = N(tsample)
        F = np.array([f(ti) for ti in tsample], dtype="float64")
        if BCvals is None:
            return np.linalg.lstsq(Lt.T, F, rcond=None)[0]
        mask = np.isnan(BCvals)
        BCvals[mask] = 0
        F -= Lt.T @ BCvals
        BCvals[mask] = 0
        nunkknown = np.sum(mask)
        indexs = np.zeros((nunkknown, 1), dtype="int32")
        k = 0
        for i in range(N.npts):
            if mask[i]:
                indexs[k, 0] = i
                k += 1
        
        B = np.zeros(nunkknown, dtype="float64")
        A = np.zeros((nunkknown, nunkknown), dtype="float64")
        for ka, (ia, ) in enumerate(indexs):
            B[ka] = Lt[ia] @ F
            for kb, (ib, ) in enumerate(indexs):
                A[ka, kb] = Lt[ia] @ Lt[ib]
        solution = np.linalg.solve(A, B)
        finalresult = np.copy(BCvals)
        for a, (ia, ) in enumerate(indexs):
            finalresult[ia] = solution[a]
        BCvals[mask].fill(np.nan)
        return finalresult

    @staticmethod
    def spline_surface(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, f: Callable[[float, float], float], BCvals: np.ndarray = None) -> np.array:
        nx, ny = Nx.npts, Ny.npts
        px, py = Nx.degree, Ny.degree
        xsample = []
        ysample = []
        xknots = Nx.knotvector.knots
        yknots = Ny.knotvector.knots
        ndivx = int(np.ceil(1/(1-px/nx)))
        ndivy = int(np.ceil(1/(1-py/ny)))
        for ta, tb in zip(xknots[:-1], xknots[1:]):
            xsample.extend(np.linspace(ta, tb, ndivx, endpoint=False))
        for ta, tb in zip(yknots[:-1], yknots[1:]):
            ysample.extend(np.linspace(ta, tb, ndivy, endpoint=False))
        xsample.append(1)
        xsample = np.array(xsample)
        ysample.append(1)
        ysample = np.array(ysample)

        nxs, nys = len(xsample), len(ysample)
        F = np.zeros((nxs, nys), dtype="float64")
        for i, xi in enumerate(xsample):
            for j, yj in enumerate(ysample):
                F[i, j] = f(xi, yj)
        
        Lx = Nx(xsample)
        Ly = Ny(ysample)
        Kx = np.linalg.inv(Lx @ Lx.T) @ Lx
        Ky = np.linalg.inv(Ly @ Ly.T) @ Ly
        if BCvals is None:
            return Kx @ F @ Ky.T
        mask = np.isnan(BCvals)
        BCvals[mask] = 0
        F -= Lx.T @ BCvals @ Ly
        BCvals[mask] = np.nan
        nunkknown = np.sum(mask)
        indexs = np.zeros((nunkknown, 2), dtype="int32")
        k = 0
        for i in range(nx):
            for j in range(ny):
                if mask[i, j]:
                    indexs[k] = i, j
                    k += 1
        B = np.zeros(nunkknown, dtype="float64")
        A = np.zeros((nunkknown, nunkknown), dtype="float64")
        for ka, (ia, ja) in enumerate(indexs):
            B[ka] = Lx[ia] @ F @ Ly[ja].T
            for kb, (ib, jb) in enumerate(indexs):
                A[ka, kb] = (Lx[ia] @ Lx[ib]) * (Ly[ja] @ Ly[jb])
        solution = np.linalg.solve(A, B)
        finalresult = np.copy(BCvals)
        for a, (ia, ja) in enumerate(indexs):
            finalresult[ia, ja] = solution[a]
        BCvals[mask].fill(np.nan)
        return finalresult

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

def getT1(N: nurbs.SplineBaseFunction, a: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Ha = np.zeros(n, dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Ma = getM1(N, k, a)
        Ha[k-a:k+1] += (U[k+1]-U[k])*Ma
    return Ha

def getT2(N: nurbs.SplineBaseFunction, a: int, b: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Hab = np.zeros((n, n), dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Mab = getM2(N, k, a, b)
        Hab[k-a:k+1, k-b:k+1] += (U[k+1]-U[k])*Mab
    return Hab

def getT3(N: nurbs.SplineBaseFunction, a: int, b:int, c: int):
    U = N.knotvector
    p, n = N.degree, N.npts
    Habc = np.zeros((n, n, n), dtype="float64")
    for k in range(p, n):
        if U[k+1] == U[k]:
            continue
        Mabc = getM3(N, k, a, b, c)
        Habc[k-a:k+1, k-b:k+1, k-c:k+1] += (U[k+1]-U[k])*Mabc
    return Habc


def getT(N: nurbs.SplineBaseFunction, *args: Tuple[int]):
    if len(args) == 1:
        return getT1(N, args[0])
    if len(args) == 2:
        return getT2(N, args[0], args[1])
    if len(args) == 3:
        return getT3(N, args[0], args[1], args[2])
    raise ValueError

def getH1(N: nurbs.SplineBaseFunction, a: int):
    n, p, U = N.npts, N.degree, N.knotvector
    if a > p:
        return np.zeros(n, dtype="float64")
    D = {}
    for i in range(1, p+1):
        D[i] = getD(i, U)
    termo = getT(N, p-a)
    for i in range(p+1-a, p+1):
        termo = D[i] @ termo
    return termo

def getH2(N: nurbs.SplineBaseFunction, a: int, b: int):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] dx
    Exemplos:
        getH(N, 0, 0) -> I = int_0^1 [Nx] x [Nx] dx
        getH(N, 1, 0) -> I = int_0^1 [Nx'] x [Nx] dx
        getH(N, 0, 1) -> I = int_0^1 [Nx] x [Nx'] dx
        getH(N, 1, 1) -> I = int_0^1 [Nx'] x [Nx'] dx
    """
    n, p, U = N.npts, N.degree, N.knotvector
    maxderoriginal = max(a, b)
    maxderpartes = int(np.ceil(np.mean([a, b])))
    D = {}
    for i in range(max(1, min(p+1-a, p+1-b)), p+1):
        D[i] = getD(i, U)
    if maxderpartes == maxderoriginal:
        termo = getT(N, p-a, p-b)
        for i in range(p+1-a, p+1):
            termo = D[i] @ termo
        for i in range(p+1-b, p+1):
            termo = termo @ D[i].T
        return termo
    
    if b < a:
        return np.transpose(getH2(N, b, a))
    for i in range(1, min(p+1-a, p+1-b)):
        D[i] = getD(i, U)
    if a == 0 and b == 2:
        termo = np.tensordot(N[:, p](1), D[p] @ N[:, p-1](1), axes=0)
        termo -= np.tensordot(N[:, p](0), D[p] @ N[:, p-1](0), axes=0)
        termo -= getH2(N, 1, 1)
        return termo
    elif a == 0 and b == 3:
        termo = np.tensordot(N[:, p](1), D[p] @ D[p-1] @ N[:, p-2](1), axes=0)
        termo -= np.tensordot(N[:, p](0), D[p] @ D[p-1] @ N[:, p-2](0), axes=0)
        termo -= getH2(N, 1, 2)
        return termo
    elif a == 0 and b == 4:
        termo = 0
        if p > 2:
            termo += np.tensordot(N(1), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](1), axes=0)
            termo -= np.tensordot(N(0), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](0), axes=0)
        termo -= getH(N, 1, 3)
        return termo
    elif a == 1 and b == 3:
        termo = 0
        termo -= np.tensordot(D[p] @ N[:, p-1](1), D[p] @ D[p-1] @ N[:, p-2](1), axes=0)
        termo += np.tensordot(D[p] @ N[:, p-1](0), D[p] @ D[p-1] @ N[:, p-2](0), axes=0)
        termo -= getH(N, 2, 2)
        return termo
    errormsg = f"Nao pude resolver: (a, b) = ({a}, {b})"
    raise ValueError(errormsg)

def getH3(N: nurbs.SplineBaseFunction, a: int, b: int, c: int):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] x [Nx^(c)] dx
    Exemplos:
        getH(N, 0, 0, 0) -> I = int_0^1 [Nx] x [Nx] x [Nx] dx
        getH(N, 0, 0, 1) -> I = int_0^1 [Nx] x [Nx] x [Nx'] dx
        getH(N, 0, 1, 0) -> I = int_0^1 [Nx] x [Nx'] x [Nx] dx
        getH(N, 2, 0, 0) -> I = int_0^1 [Nx''] x [Nx] x [Nx] dx
    """
    n, p, U = N.npts, N.degree, N.knotvector
    maxderoriginal = max(a, b, c)
    maxderpartes = int(np.ceil(np.mean([a, b, c])))
    D = {}
    for i in range(max(1, min(p+1-a, p+1-b, p+1-c)), p+1):
        D[i] = getD(i, U)
    if maxderpartes == maxderoriginal:
        termo = getT(N, p-a, p-b, p-c)
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
        if a == 0 and b == 0 and c == 2:
            termo = np.einsum("kc,ijc->ijk", D[p] @ D[p-1], getT(N, p, p, p-2))
            return termo
        if a == 0 and b == 0 and c == 3:
            termo = np.einsum("i,j,k->ijk", N[:, p](1), N[:, p](1), D[p] @ D[p-1] @ N[:, p-2](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), N[:, p](0), D[p] @ D[p-1] @ N[:, p-2](0))
            termo -= getH(N, 1, 0, 2)
            termo -= getH(N, 0, 1, 2)
            return termo
        if a == 0 and b == 0 and c == 4:
            termo = np.einsum("i,j,k->ijk", N[:, p](1), N[:, p](1), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), N[:, p](0), D[p] @ D[p-1] @ D[p-2] @ N[:, p-3](0))
            termo -= getH(N, 1, 0, 3)
            termo -= getH(N, 0, 1, 3)
            return termo
        if a == 0 and b == 0 and c == 5:
            termo = np.einsum("i,j,k->ijk", N[:, p](1), N[:, p](1), D[p] @ D[p-1] @ D[p-2] @ D[p-3] @ N[:, p-4](1))
            termo -= np.einsum("i,j,k->ijk", N[:, p](0), N[:, p](0), D[p] @ D[p-1] @ D[p-2] @ D[p-3] @ N[:, p-4](0))
            termo -= getH(N, 1, 0, 4)
            termo -= getH(N, 0, 1, 4)
            return termo
        if a == 0 and b == 1 and c == 2:
            # termo = np.einsum("i,j,k->ijk", N[:, p](1), D[p] @ N[:, p-1](1), D[p] @ N[:, p-1](1))
            # termo -= np.einsum("i,j,k->ijk", N[:, p](0), D[p] @ N[:, p-1](0), D[p] @ N[:, p-1](0))
            # termo -= getH3(N, 1, 1, 1)
            # termo -= np.einsum("jb,kc,ibc->ijk", D[p] @ D[p-1], D[p], getT(N, p, p-2, p-1))
            termo = np.einsum("jb,kc,ibc->ijk", D[p], D[p] @ D[p-1], getT(N, p, p-1, p-2))
            return termo
        errormsg = f"Nao pude resolver: (a, b, c) = ({a}, {b}, {c})"
        raise ValueError(errormsg)
    termo = getH3(N, a0, b0, c0)
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
        termo = getH3(N, a0, b0, c0)
        # First, we put a in its place
        if a == b0:
            np.swapaxes(termo, 0, 1)
        elif a == c0:
            np.swapaxes(termo, 0, 2)
        np.swapaxes(termo, 1, 2)
        return termo
        

def getH(N: nurbs.SplineBaseFunction, *args: Tuple[int]):
    """
    Retorna um valor de [I], a integral de
    I = int_0^1 [Nx^(a)] x [Nx^(b)] x ... x [Nx^(z)] dx
    em que args = (a, b, ..., z)
    Exemplos:
        getH(N, 0) -> I = int_0^1 [Nx] dx
        getH(N, 0, 0) -> I = int_0^1 [Nx] x [Nx] dx
        getH(N, 0, 0, 0) -> I = int_0^1 [Nx] x [Nx] x [Nx] dx
        getH(N, 1) -> I = int_0^1 [Nx'] dx
        getH(N, 2) -> I = int_0^1 [Nx''] dx
        getH(N, 1, 0) -> I = int_0^1 [Nx'] x [Nx] dx
        getH(N, 0, 1) -> I = int_0^1 [Nx] x [Nx'] dx
        getH(N, 1, 1) -> I = int_0^1 [Nx'] x [Nx'] dx
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
        return getH1(N, args[0])
    elif ndim == 2:
        return getH2(N, *args)
    elif ndim == 3:
        return getH3(N, *args)
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

if __name__ == "__main__":
    
    p = np.random.randint(1, 7)
    n = np.random.randint(p+1, p+10)
    U = nurbs.GeneratorKnotVector.random(p, n)
    N = nurbs.SplineBaseFunction(U)
    

    print("Ti = ")
    for i in range(p+1):
        Ti = getT1(N, i)
        assert np.abs(np.sum(Ti) - 1) < 1e-6
    
    print("Tij = ")
    for i in range(p):
        for j in range(p):
            Tij = getT2(N, i, j)
            assert np.abs(np.sum(Tij) - 1) < 1e-6

    print("Tijk = ")
    for i in range(p):
        for j in range(p):
            for k in range(p):
                Tijk = getT3(N, i, j, k)
                assert np.abs(np.sum(Tijk) - 1) < 1e-6

    print("Hi = ")
    for i in range(p+1):
        Hi = getH1(N, i)
        print("    " + str(np.sum(Hi)))

    print("Hij = ")
    for i in range(p):
        for j in range(p):
            Hij = getH2(N, i, j)
            print("    " + str(np.sum(Hij)))

    print("Hijk = ")
    for i in range(p):
        for j in range(p):
            for k in range(p):
                Hijk = getH3(N, i, j, k)
                print("    " + str(np.sum(Hijk)))

    # print("Tij = ")
    # for i in range(p):
    #     for j in range(p):
    #         Tij = getT2(N, i, j)
    #         print("    " + str(np.sum(Tij)))

    # print("Tijk = ")
    # for i in range(p):
    #     for j in range(p):
    #         for k in range(p):
    #             Tijk = getT3(N, i, j, k)
    #             print("    " + str(np.sum(Tijk)))



    # xplot = np.linspace(0, 1, 129)
    # for i in range(n):
    #     plt.plot(xplot, D[p][i] @ N[:, p-1](xplot), label=r"$N'_{%d,%d}$"%(i,p))
    # plt.grid()
    # plt.legend()
    # plt.show()