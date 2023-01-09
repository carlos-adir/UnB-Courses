from compmec import nurbs
import math
import numpy as np
from typing import Callable, Tuple

def getMatrix(y: int, z: int, k: int, N: nurbs.SplineBaseFunction):
    """
    Dado o par (y, z) essa funcao retorna o valor de [M_yz]
    Em que 
        (u_{k+1}-u_{k}) * [M_yz] = int_{u_k}^{u_{k+1}} [N_y] x [N_z] du
    """
    if y == 0 and z == 0:
        return np.ones((1, 1))

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