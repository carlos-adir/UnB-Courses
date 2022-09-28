import numpy as np
import sympy as sp
from typing import Iterable, Any, Tuple, Optional, Dict
from matplotlib import pyplot as plt
from matplotlib import cm
import os, sys
from ellipse import Ellipse        

def readlines(filename: str, header: int = 0) -> np.ndarray:
    with open(filename, "r") as file:
        alllines = file.readlines()
    for i in range(header):
        alllines.pop(0)
    if len(alllines[-1]) < 5:
        alllines.pop(-1)
    for i, line in enumerate(alllines):
        alllines[i] = line.replace("\n", "").split("\t")
        for j, val in enumerate(alllines[i]):    
            alllines[i][j] = float(val)
    return np.array(alllines)

def read_all_files() -> Dict[str, Dict[int, np.ndarray]]:
    CURRENTFOLDER = os.path.dirname(__file__)
    folders = {"massa1/": [1, 3, 6, 9, 11, 12, 15, 18, 21],
               "massa2/": [1, 3, 6, 9, 12, 14, 15, 18, 21],
               "massa3/": [1, 3, 6, 9, 12, 15, 16, 17, 18, 19, 21, 22, 23, 25, 28, 30],
               "massa4/": [1, 5, 9, 13, 15, 16, 17, 18, 19, 20, 21, 23, 27, 29]}
    suffixfilename = "Hz_tmp_001.txt"
    alldatavalues = {} 
    for folder, frequencies in folders.items():
        alldatavalues[folder] = {}
        for frequency in frequencies:
            completefilename = os.path.join(CURRENTFOLDER,  folder, str(frequency) + suffixfilename)
            data = readlines(completefilename, header=23)
            alldatavalues[folder][frequency] = data
    return alldatavalues

def get_ddx_over_f(ddxvals: Iterable[float], fvals: Iterable[float], residual: Optional[bool]=False):
    """
    This function receives a set of acceleration values and force vals
    It makes a ellipse regression and returns the value of
        ddx/f = H(wi)/m
    as a cmplex
    """
    ellipse = Ellipse()
    ellipse.fit(ddxvals, fvals)
    x_med, y_med = ellipse.center
    x_amp, y_amp = ellipse.amplitude
    x_ref, y_max = ellipse.get_point_ymax()
    gain = x_amp/y_amp
    phase = np.arccos((x_ref-x_med)/x_amp)
    retorno = gain*np.cos(phase) + 1j*gain*np.sin(phase)
    if not residual:
        return retorno
    distances = ellipse.distance(ddxvals, fvals)
    resid = np.mean(distances)
    return retorno, resid
    
def fit_curve(wvals: Iterable[float], Hoverm: Iterable[complex], weight: Optional[Iterable[float]]=None):
    """
    Finds the best value for (m), (c), (k) using the (wvals), (gain) and (phase)
    """
    if weight is None:
        weight = np.ones(len(wvals))
    wvals = np.array(wvals)
    gain = np.abs(Hoverm)
    x = np.real(Hoverm)
    y = np.imag(Hoverm)
    
    g2 = gain**2
    ws2 = wvals**2
    ws3 = ws2*wvals
    ws4 = ws2**2
    sumg2 = np.sum(weight*g2)
    sumw2g2 = np.sum(weight*ws2*g2)
    sumw4g2 = np.sum(weight*ws4*g2)
    sumxw2 = np.sum(weight*x*ws2)
    sumxw4 = np.sum(weight*x*ws4)

    c = np.sum(weight*y*ws3)/sumw2g2
    M = [[sumw4g2, -sumw2g2],
         [-sumw2g2, sumg2]]
    B = [sumxw4, -sumxw2]
    m, k = np.linalg.solve(M, B)
    return m, c, k


def H(xi: float, r: Iterable[float]):
    return -r**2 / (1-r**2 +2j*xi*r)

def main():
    alldatavalues = read_all_files()
    for nmass, (folder, datafolder) in enumerate(alldatavalues.items()):
        if nmass == 0:
            xiold = 0.0138
            wnold = 70.34
            wntheo = 64.42
        elif nmass == 1:
            xiold = 0.0173135
            wnold = 107.033659497
            wntheo = 87.26
        elif nmass == 2:
            xiold = 0.0204799371381206
            wnold = 124.20297799834222
            wntheo = 124.5
        else:
            xiold = 0.020171393290422755
            wnold = 142.12969140933308
            wntheo = 152.5

        frequencies = np.array(list(datafolder.keys()))
        wdots = 2*np.pi*frequencies
        Hoverm = np.zeros(len(frequencies), dtype="complex128")
        for i, (frequency, data) in enumerate(datafolder.items()):
            xvalues = data[:, 1]  # Acceleration
            yvalues = data[:, 2]  # Force
            Hoverm[i] = get_ddx_over_f(xvalues, yvalues)
            
        msupo, csupo, ksupo = fit_curve(wdots, Hoverm)
        wnsupo = np.sqrt(ksupo/msupo)
        xisupo = csupo/(2*np.sqrt(ksupo*msupo))

        angle = 0.85*(np.pi/2)  # To fit interpolation
        uvals = np.linspace(0, angle, 1025)
        rplot1 = np.exp(-np.tan(uvals)*np.log(wnsupo/min(wdots))/np.tan(angle))
        rplot2 = np.exp(np.tan(uvals)*np.log(max(wdots)/wnsupo)/np.tan(angle))
        rplot3 = np.linspace(0, min(rplot1), 129, endpoint=False)
        rplot4 = np.linspace(max(rplot2), 200, 129)
        rplot5 = np.exp(-np.tan(uvals)*np.log(wnold/min(wdots))/np.tan(angle))
        rplot6 = np.exp(np.tan(uvals)*np.log(max(wdots)/wnold)/np.tan(angle))
        rplot = np.concatenate([rplot1, rplot2, rplot3, rplot4, rplot5, rplot6])
        rplot.sort()
        wplot = wnsupo*rplot
        Hplot = H(xisupo, rplot)
        Hold = H(xiold, wplot/wnold)
        print("Fitting values: " + str(folder))
        print("    m = %.6f" % msupo)
        print("    c = %.6f" % csupo)
        print("    k = %.6f" % ksupo)
        print("   wn = %.6f" % wnsupo)
        print("   xi = %.6f" % xisupo)

        plt.figure(figsize=(20, 5))
        plt.plot(wplot, np.abs(Hold)/msupo, color="r", ls="dotted", label=r"exper 1")
        plt.plot(wplot, np.abs(Hplot)/msupo, color="b", ls="dotted", label=r"estimated")
        plt.scatter(wdots, np.abs(Hoverm), color="y", marker="o", edgecolors="k", label=r"readed")
        plt.xlabel(r"Frequencies $\omega$ rad/s")
        plt.ylabel(r"Gain of $\ddot{x}/f$")
        plt.axhline(y=1/msupo, color="g", ls="dashed", label=r"$1/m$")
        plt.axvline(x=wntheo, ls="dashed", color="m", label=r"$\left[\omega_{n}\right]_{theo}$")
        plt.legend()
        plt.xlim((0, 200))
        plt.grid()
        plt.title("Gain for " + str(folder))
        plt.savefig("Gain" + folder.replace("/", "") + ".png")
        
        plt.figure(figsize=(20, 5))
        plt.plot(wplot, np.angle(Hold), color="r", ls="dotted", label=r"exper 1")
        plt.plot(wplot, np.angle(Hplot), color="b", ls="dotted", label=r"estimated")
        plt.scatter(wdots, np.angle(Hoverm), color="y", marker="o", edgecolors="k", label=r"readed")
        plt.xlabel(r"Frequencies $\omega$ rad/s")
        plt.ylabel(r"Phase of $\ddot{x}/f$")
        plt.axvline(x=wntheo, ls="dashed", color="m", label=r"$\left[\omega_{n}\right]_{theo}$")
        plt.axhline(y=0, color="g", ls="dashed")
        plt.axhline(y=np.pi/2, color="g", ls="dashed")
        plt.axhline(y=np.pi, color="g", ls="dashed")
        plt.legend()
        plt.xlim((0, 200))
        plt.grid()
        plt.title("Phase Diagram for " + str(folder))
        plt.savefig("Phase" + folder.replace("/", "") + ".png")

        plt.figure(figsize=(20, 5))
        plt.plot(wplot, np.real(Hold)/msupo, color="r", ls="dotted", label=r"exper 1")
        plt.plot(wplot, np.real(Hplot)/msupo, color="b", ls="dotted", label=r"estimated")
        plt.scatter(wdots, np.real(Hoverm), color="y", marker="o", edgecolors="k", label=r"readed")
        plt.xlabel(r"Frequencies $\omega$ rad/s")
        plt.ylabel(r"Real part of $\ddot{x}/f$")
        plt.axhline(y=0, color="g", ls="dashed")
        plt.axvline(x=wntheo, ls="dashed", color="m", label=r"$\left[\omega_{n}\right]_{theo}$")
        plt.legend()
        plt.xlim((0, 200))
        plt.grid()
        plt.title("Real part for " + str(folder))
        plt.savefig("Real" + folder.replace("/", "") + ".png")

        plt.figure(figsize=(20, 5))
        plt.plot(wplot, np.imag(Hold)/msupo, color="r", ls="dotted", label=r"exper 1")
        plt.plot(wplot, np.imag(Hplot)/msupo, color="b", ls="dotted", label=r"estimated")
        plt.scatter(wdots, np.imag(Hoverm), color="y", marker="o", edgecolors="k", label=r"readed")
        plt.xlabel(r"Frequencies $\omega$ rad/s")
        plt.ylabel(r"Imaginary part of $\ddot{x}/f$")
        plt.axvline(x=wntheo, ls="dashed", color="m", label=r"$\left[\omega_{n}\right]_{theo}$")
        plt.axhline(y=0, color="g", ls="dashed")
        plt.xlim((0, 200))
        plt.legend()
        plt.grid()
        plt.title("Imaginary part for " + str(folder))
        plt.savefig("Imag" + folder.replace("/", "") + ".png")

        plt.figure(figsize=(6, 6))
        plt.plot(np.real(Hold)/msupo, np.imag(Hold)/msupo, color="r", ls="dotted", label=r"exper 1")
        plt.plot(np.real(Hplot)/msupo, np.imag(Hplot)/msupo, color="b", ls="dotted", label=r"estimated")
        plt.scatter(np.real(Hoverm), np.imag(Hoverm), color="y", marker="o", edgecolors="k", label=r"readed")
        plt.axvline(x=1/msupo, ls="dashed", color="g", label=r"$1/m$")
        plt.xlabel(r"Real part $\ddot{x}/f$")
        plt.ylabel(r"Imaginary part $\ddot{x}/f$")
        plt.title("Complex plane diagram for " + str(folder))
        plt.legend()
        plt.grid()
        plt.gca().axis("equal")
        plt.savefig("ComplexPlane" + folder.replace("/", "") + ".png")
        # plt.show()


if __name__ == "__main__":
    main()
    