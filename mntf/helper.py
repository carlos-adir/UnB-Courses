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
    if not (a <= b and b <= c):
        raise ValueError(f"a, b, c must be in order: {a}, {b}, {c}")
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
    

def solve_system(A: np.ndarray, B: np.ndarray, X: np.ndarray, X0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve o sistema A * X = B
    Recebe 
        ```A```: Matriz de tamanho (n1, n2, n1, n2)
        ```B```: Matriz de tamanho (n1, n2)
        ```X```: Matriz de tamanho (n1, n2), com valores ```nan``` dentro e com condicoes de contorno
    Se X0 for dado, eh usado um metodo iterativo
    """
    
    if B.shape != X.shape:
        raise ValueError(f"B.shape = {B.shape} != {X.shape} = X.shape")
    if X0 is not None:
        if X0.shape != X.shape:
            raise ValueError(f"X0.shape = {X0.shape} != {X.shape} = X.shape")
    if A.ndim != 2*B.ndim:
        raise ValueError("A.ndim = {A.ndim} != 2*{B.ndim} = 2*B.ndim")
    if not np.any(np.isnan(X)):
        raise ValueError("At least one unknown must be given! All values of X are known")

    Xoriginal = np.copy(X)
    ns = B.shape
    ndim = len(ns)
    ntot = np.prod(ns)
    Aexp = np.zeros((ntot, ntot), dtype="float64")
    Bexp = np.zeros(ntot, dtype="float64")
    if ndim == 1:
        Aexp = np.copy(A)
        Bexp = np.copy(B)
        Xexp = np.copy(X)
        if X0 is not None:
            X0exp = np.copy(X0)
    elif ndim == 2:
        Aexp = np.zeros((ntot, ntot), dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                Aexp[i*ns[1]+j, :] = A[i, :, j, :].reshape(ntot)
        Bexp = np.copy(B).reshape(ntot)
        Xexp = np.copy(X).reshape(ntot)
        if X0 is not None:
            X0exp = np.copy(X0).reshape(ntot)
    elif ndim == 3:
        Aexp = np.zeros((ntot, ntot), dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                for k in range(ns[2]):
                    Aexp[i*ns[2]*ns[1]+j*ns[2]+k, :] = A[i, :, j, :, k].reshape(ntot)
        Bexp = np.copy(B).reshape(ntot)
        Xexp = np.copy(X).reshape(ntot)
        if X0 is not None:
            X0exp = np.copy(X0).reshape(ntot)
    mexp = np.isnan(Xexp)
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



def main_test_solve_system():
    aproximacao = True

    ntests = 200
    for kkk in tqdm(range(ntests)):
        n1 = np.random.randint(4, 9)
        A = get_random_matrix_definite_positive(n1)
        Xgood = np.random.rand(n1)
        B = np.einsum("ij,j->i", A, Xgood)
        BCs = np.zeros(n1, dtype="bool")
        allchoices = [a for a in range(n1)]
        numbers = default_rng().choice(allchoices, size=np.random.randint(1, n1), replace=False)
        for i in numbers:
            BCs[i] = True
        Xboundary = np.empty(n1)
        Xboundary.fill(np.nan)
        Xboundary[BCs] = Xgood[BCs]
        if aproximacao:
            Xinit = Xgood+0.01*np.random.rand(n1)
            Xinit[BCs] = Xgood[BCs]
            X1, _ = solve_system(A, B, Xboundary, Xinit)
        else:
            X1, _ = solve_system(A, B, Xboundary)
        mexp = np.isnan(Xboundary)
        np.testing.assert_almost_equal(X1[~mexp], Xgood[~mexp])
        np.testing.assert_almost_equal(X1[mexp], Xgood[mexp])

    for kkk in tqdm(range(ntests)):
        ns = np.random.randint(4, 9, size=2)
        ntot = np.prod(ns)

        BCs = np.zeros(ns, dtype="bool")
        allchoices = []
        for a in range(ns[0]):
            for b in range(ns[1]):
                allchoices.append((a, b))
        numbers = default_rng().choice(allchoices, size=np.random.randint(1, ntot), replace=False)
        for i, j in numbers:
            BCs[i, j] = True
        
        Aexp = get_random_matrix_definite_positive(ntot)
        Aexp = np.around(Aexp, 3)
        A = np.zeros((ns[0], ns[0], ns[1], ns[1]), dtype="float64")
        Xgood = np.around(np.random.rand(*ns), 1)
        B = np.zeros((ns[0], ns[1]), dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                A[i, :, j, :] = Aexp[i*ns[1]+j, :].reshape(ns)
        for i in range(ns[0]):
            for j in range(ns[1]):
                B[i, j] += np.tensordot(A[i, :, j, :], Xgood, axes=2)
        Xexpgood = np.zeros((ntot), dtype="float64")
        Bexp = np.zeros(ntot, dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                Xexpgood[i*ns[1]+j] = Xgood[i, j]
                Bexp[i*ns[1]+j] = B[i, j]
        
        Xboundary = np.empty(ns)
        Xboundary.fill(np.nan)
        Xboundary[BCs] = Xgood[BCs]

        Aexptest = np.zeros((ntot, ntot), dtype="float64")
        Bexptest = B.reshape(ntot)
        np.testing.assert_almost_equal(Bexptest, Bexp)
        
        assert np.all( np.abs(Aexp @ Xexpgood - Bexp) < 1e-6)

        if aproximacao:
            Xinit = Xgood+0.01*np.random.rand(*ns)
            Xinit[BCs] = Xboundary[BCs]
            X1, _ = solve_system(A, B, Xboundary, Xinit)
        else:
            X1, _ = solve_system(A, B, Xboundary)
        
        mexp = np.isnan(Xboundary)
        np.testing.assert_almost_equal(X1[~mexp], Xgood[~mexp])
        np.testing.assert_almost_equal(X1[mexp], Xgood[mexp])
        # np.testing.assert_almost_equal(X1, Xgood)

    for kkk in tqdm(range(ntests)):
        ns = np.random.randint(4, 9, size=3)
        ntot = np.prod(ns)

        BCs = np.zeros(ns, dtype="bool")
        allchoices = []
        for a in range(ns[0]):
            for b in range(ns[1]):
                for c in range(ns[2]):
                    allchoices.append((a, b, c))
        numbers = default_rng().choice(allchoices, size=np.random.randint(1, ntot), replace=False)
        for i, j, k in numbers:
            BCs[i, j, k] = True
        
        Aexp = get_random_matrix_definite_positive(ntot)
        Aexp = np.around(Aexp, 3)
        A = np.zeros((ns[0], ns[0], ns[1], ns[1], ns[2], ns[2]), dtype="float64")
        Xgood = np.around(np.random.rand(*ns), 1)
        B = np.zeros(ns, dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                for k in range(ns[2]):
                    A[i, :, j, :, k, :] = Aexp[i*ns[1]*ns[2]+j*ns[2]+k, :].reshape(ns)
        for i in range(ns[0]):
            for j in range(ns[1]):
                for k in range(ns[2]):
                    B[i, j, k] += np.tensordot(A[i, :, j, :, k, :], Xgood, axes=3)
        Xexpgood = np.zeros((ntot), dtype="float64")
        Bexp = np.zeros(ntot, dtype="float64")
        for i in range(ns[0]):
            for j in range(ns[1]):
                for k in range(ns[2]):
                    Xexpgood[i*ns[1]*ns[2]+j*ns[2]+k] = Xgood[i, j, k]
                    Bexp[i*ns[1]*ns[2]+j*ns[2]+k] = B[i, j, k]
        
        Xboundary = np.empty(ns)
        Xboundary.fill(np.nan)
        Xboundary[BCs] = Xgood[BCs]

        Aexptest = np.zeros((ntot, ntot), dtype="float64")
        Bexptest = B.reshape(ntot)
        np.testing.assert_almost_equal(Bexptest, Bexp)
        
        assert np.all( np.abs(Aexp @ Xexpgood - Bexp) < 1e-6)

        if aproximacao:
            Xinit = Xgood+0.1*np.random.rand(*ns)
            Xinit[BCs] = Xboundary[BCs]
            X1, _ = solve_system(A, B, Xboundary, Xinit)
        else:
            X1, _ = solve_system(A, B, Xboundary)
        mexp = np.isnan(Xboundary)
        np.testing.assert_almost_equal(X1[~mexp], Xgood[~mexp])
        np.testing.assert_almost_equal(X1[mexp], Xgood[mexp])


def test_matrix():
    n, p = 3, 2
    U = nurbs.GeneratorKnotVector.uniform(p, n)
    N = nurbs.SplineBaseFunction(U)
    H = getH(N, 2, 1)
if __name__ == "__main__":
    test_matrix()