
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

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
    for i, indsi in enumerate(indexskno):  # find B
        indexsmat[::2] = indsi
        for j, indsj in enumerate(indexskno):  # Use X
            indexsmat[1::2] = indsj
            iBX[tuple(indexsmat)] = F[i, j]
        for j, indsj in enumerate(indexsnan):  # Use B
            indexsmat[1::2] = indsj
            iBB[tuple(indexsmat)] = G[i, j]
    return ((iXX, iXB), (iBX, iBB))

def eigenvalues(A: np.ndarray, mask: np.ndarray = None):
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
    Retorna os autovalores de [Auu]
    """
    ns = A.shape[::2]
    if mask is None:
        mask = np.ones(ns, dtype="bool")
    elif A.ndim != 2*mask.ndim:
        raise ValueError(f"A.ndim = {A.ndim} != 2*{mask.ndim} = 2*X.ndim")
    elif np.prod(A.shape) != np.prod(mask.shape)**2:
        raise ValueError(f"A.shape = {A.shape} != 2*{mask.shape} = 2*X.shape")
    if not np.any(mask):
        raise ValueError(f"At least one unknown must be given! All values of X are known")

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
    eig = np.linalg.eigvals(Auu)
    return eig

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
    for ndim in [1, 2, 3]:
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
    for ndim in [1, 2, 3]:
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