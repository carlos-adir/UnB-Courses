import numpy as np
import sympy as sp
from typing import Iterable, Any, Tuple
from matplotlib import pyplot as plt

def solution_underdamped(xi:float, wn:float, t: Any, x0:float, v0:float):
    mu = np.sqrt(1-xi**2)
    x = sp.exp(-xi*wn*t) * (x0 * sp.cos(mu*wn*t) + (xi*x0/mu + v0/(mu*wn)) * sp.sin(mu*wn*t))
    return x

def solution_x(xi: float, wn: float, x0: float, v0: float):
    t = sp.symbols("t")
    if 0 < xi < 1:
        x = solution_underdamped(xi, wn, t, x0, v0)
    else:
        raise Exception("The value of xi must be in (0, 1). Received %.3f" % xi)
    return sp.lambdify(t, x)

def solution_v(xi:float, wn:float, x0:float, v0:float):
    t = sp.symbols("t")
    if 0 < xi < 1:
        x = solution_underdamped(xi, wn, t, x0, v0)
    else:
        raise Exception("The value of xi must be in (0, 1). Received %.3f" % xi)
    v = sp.diff(x, t)
    return sp.lambdify(t, v)

def getMB(tvals: np.ndarray, xvals: np.ndarray, Xj: np.ndarray):
    # Given the Xj vector
    #     Xj = (Aj, Bj, wxj, wdj),
    # and the points (ti, xi),
    #     xi = xvals[i]
    #     ti = tvals[i]
    # It returns a [M] and [B] such that
    #     [M] = [nabla nabla J], shape = (4 x 4)
    #     [B] = [nabla J], shape = (4) 
    n = len(xvals)
    m = len(Xj)
    A, B, wx, wd = Xj
    exp = np.exp(wx*tvals)
    cos = np.cos(wd*tvals)
    sin = np.sin(wd*tvals)
    expcos = exp * cos
    expsin = exp * sin
    
    L = A * expcos + B * expsin
    dL = np.zeros((m, n), dtype="float64")
    ddL = np.zeros((m, m, n), dtype="float64")
    
    dL[0] = expcos
    dL[1] = expsin
    dL[2] = tvals * L
    dL[3] = tvals * (-A*expsin + B*expcos)
    
    ddL[0, 2] = ddL[2, 0] = tvals * expcos
    ddL[0, 3] = ddL[3, 0] = -tvals * expsin
    ddL[1, 2] = ddL[2, 1] = -ddL[0, 3]
    ddL[1, 3] = ddL[3, 1] = ddL[0, 2]
    ddL[2, 3] = ddL[3, 2] = tvals * dL[3]
    ddL[2, 2] = tvals * dL[2]
    ddL[3, 3] = - ddL[2, 2]
    
    weight = np.ones(len(tvals))
    
    W = np.diag(weight)
    M = dL @ W @ dL.T - ddL @ W @ xvals + ddL @ W @ L
    B = dL @ W @ L - dL @ W @ xvals
    return M, B

def find_roots(x: Iterable[float], qtt: float = 3):
    roots = []
    absx = np.abs(x)
    i = 0
    while len(roots) < qtt:
        while x[i] * x[i+1] > 0:
            i += 1
        roots.append(i)
        i = np.where(absx[i:] == np.max(absx[i:]))[0][0] + roots[-1]
    return roots

def get_initial_wxwd(tsample: Iterable[float], xsample: Iterable[float]):
    xsample = np.copy(xsample)
    tsample = np.copy(tsample)
    xfiltered = filter(xsample)
    roots = find_roots(xfiltered, 10)
    xabs = np.abs(xsample)
    nroots = len(roots)-1
    L = np.ones((nroots-2, 2))
    y = np.ones(nroots-2)
    for i in range(nroots-2):
        ra = roots[i+1]
        rb = roots[i+2]
        mask = (xabs[ra:rb] == np.max(xabs[ra:rb]))
        indexmax = np.where(mask)[0][0] + ra
        L[i, 1] = tsample[indexmax]
        y[i] = np.log(xabs[indexmax])
    params = np.linalg.lstsq(L, y, rcond=None)[0]
    wx = params[1]
    
    Td = 2 * np.mean(tsample[roots[2:]] - tsample[roots[1:-1]])
    wd = 2*np.pi/Td
    return wx, wd

def get_initial_AB(tsample: Iterable[float], xsample: Iterable[float], wx: float, wd: float):
    tsample = np.array(tsample)
    xsample = np.array(xsample)
    exp = np.exp(wx*tsample)
    L = np.array([exp*np.cos(wd*tsample), exp*np.sin(wd*tsample)])
    return np.linalg.solve(L @ L.T, L @ xsample)

def find_initial_vector(tsample:np.ndarray, xsample:np.ndarray):
    wx0, wd0 = get_initial_wxwd(tsample, xsample)
    A0, B0 = get_initial_AB(tsample, xsample, wx0, wd0)
    return np.array([A0, B0, wx0, wd0])

def transform_X2Z(X: Tuple[float]) -> Tuple[float]:
    A, B, wx, wd = X
    wn = np.sqrt(wx**2 + wd**2)
    xi = -wx/wn
    x0 = A
    v0 = B * wd - xi * wn * x0
    return xi, wn, x0, v0

def transform_Z2X(Z: Tuple[float]) -> Tuple[float]:
    xi, wn, x0, v0 = Z
    mu = np.sqrt(1-xi**2)
    wx = -xi * wn
    wd = wn * mu
    A = x0
    B = (xi/mu)*x0 + v0/wd
    return A, B, wx, wd
    

def compute_residuo(tvals: Iterable[float], xvals: Iterable[float], X):
    xi, wn, x0, v0 = transform_X2Z(X)
    xtest = solution_x(xi, wn, x0, v0)(tvals)
    return np.linalg.norm(xtest - xvals)

def filter(xvalues: np.ndarray):
    n = len(xvalues)
    b = n//200
    xfiltered = np.zeros(xvalues.shape)
    for i in range(b):
        xfiltered[i] = np.mean(xvalues[:i+1])
    for i in range(b, n-b):
        xfiltered[i] = np.mean(xvalues[i-b:i+b])
    for i in range(n-b, n):
        xfiltered[i] = np.mean(xvalues[i:])
    return xfiltered

def getXResiduoMinimal(tsample: Iterable[float], xsample: Iterable[float], ndiv = 8):
    tsample = np.array(tsample)
    xsample = np.array(xsample)
    X = find_initial_vector(tsample, xsample)
    residuals = {}
    wxs = np.linspace(0.9, 1.1, ndiv+1) * X[2]
    wds = np.linspace(0.9, 1.1, ndiv+1) * X[3]

    for wx in wxs:
        for wd in wds:
            X[2:] = wx, wd
            residuo = compute_residuo(tsample, xsample, X)
            residuals[residuo] = np.copy(X)
    return residuals[min(residuals.keys())]


def LeastSquare(tsample: Iterable[float], xsample: Iterable[float], tolerance=1e-6, nitermax = 100, verbose=False):
    tsample = np.array(tsample)
    xsample = np.array(xsample)
    X = getXResiduoMinimal(tsample, xsample)
    oldresiduo = 0
    if verbose:
        print(" ---- Begin ---- ")
    try:
        for niter in range(nitermax):
            residuo = compute_residuo(tsample, xsample, X)
            if verbose:
                print("    X%d = " % niter, X)
                print("    Z%d = " % niter, transform_X2Z(X))
                print("    Residuo = ", residuo)
            if np.abs(residuo - oldresiduo) < tolerance:
                break
            oldresiduo = residuo
            M, B = getMB(tsample, xsample, X)
            oldX = np.copy(X)
            X -= np.linalg.solve(M, B)
    except Exception as e:
        print("Least Square didn't converge")
        return oldX
    return X

def findparameters(tsample: Iterable[float], xsample: Iterable[float]):
    tsample = np.array(tsample)
    X = LeastSquare(tsample, xsample, tolerance=1e-12, verbose=True)
    Z = transform_X2Z(X)
    return Z

def readlines(filename: str) -> np.ndarray:
    with open(filename, "r") as file:
        alllines = file.readlines()
    alllines.pop(0)
    alllines.pop(-1)
    for i, line in enumerate(alllines):
        alllines[i] = line.replace("\n", "").split("\t")
        for j, val in enumerate(alllines[i]):
            alllines[i][j] = float(val)
    return np.array(alllines)


if __name__ == "__main__":
    folders = {"massa1/": [("test1-1/", 0.1)],
               "massa2/": [("test1-2/", 0.1), ("test2-2/", 0.1), ("test3-2/", 0.3)],
               "massa3/": [("test1-3/", 0.3), ("test2-3/", 0.3)],
               "massa4/": [("test1-4/", 0.3), ("test2-4/", 0.3), ("test3-4/)", 0.3)]}
    try:
        for folder, testes in folders.items():
            for teste, timecutoff in testes:
                # filename = "frq.txt"
                # filename = "mfc.txt"
                filename = "tps.txt"
                completefilename = folder + teste + filename
                print("For file: " + completefilename)
                data = readlines(completefilename)
                time = np.array(data[:, 0])
                amesured = np.array(data[:, 2])
                index = np.where(time < timecutoff)[0][-1]
                Z = findparameters(time[index:], amesured[index:])
                xi, wn, x0, v0 = Z
                print("    Parameters = ")
                print("    cutoff = %.2f" % timecutoff)
                print("        xi = %.6f" % xi)
                print("        wn = %.6f" % wn)
                print("        x0 = %.6f" % x0)
                print("        v0 = %.6f" % v0)
                print("    residu = %.6f" % compute_residuo(time[index:], amesured[index:], transform_Z2X(Z)))
    except Exception as e:
        print(e)
        aestimated = solution_x(xi, wn, x0, v0)(time)
        print("Error: ")
        # plt.plot(data[:, 0], data[:, 1], color="r", label="1")
        plt.plot(time, amesured, color="b", ls="dotted", label="Mesured")
        plt.plot(time, aestimated, color="r", label="Estimated")
        # # plt.plot(data[:, 0], data[:, 3], color="g", label="3")
        plt.legend()
        plt.title(completefilename)
        plt.xlabel("Tempo $t$ (s)")
        plt.show()
        raise e
    