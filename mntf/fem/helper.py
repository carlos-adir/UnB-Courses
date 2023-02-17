from compmec import nurbs
import math
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, Optional
from numpy.random import default_rng
np.set_printoptions(4)
from helpernurbs import *
from helperlinalg import *

def file_exists(filename: str) -> bool:
    try:
        file = open(filename, "r")
        file.close()
        return True
    except FileNotFoundError:
        return False

def print_matrix(M: np.ndarray, name: Optional[str] = None):
    assert M.ndim == 2
    if name:
        print(name + " = ")
    print(M.T[::-1])

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


    

