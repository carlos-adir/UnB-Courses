import numpy as np
import sympy as sp
from typing import Iterable, Any, Tuple, Optional, Dict
from matplotlib import pyplot as plt
import os, sys



    

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

def LinearRegression(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    """
    Given a set of points (x, y), this function returns the linear regression using
    the least square method.
    It returns the best values of A and B that (A + B*x) fits y
    """
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError("X and Y must have the same lenght!")
    if x.ndim != 1:
        raise ValueError("The X vector must be a 1D-array like")
    if y.ndim != 1:
        raise ValueError("The Y vector must be a 1D-array like")    
    L = np.array([np.ones(len(x)), np.copy(x)])
    return np.linalg.solve(L @ L.T, L @ y)

def rotate_clockwise(x: Iterable[float], y: Iterable[float], angle: float) -> np.ndarray:
    npts = len(x)
    if len(y) != npts:
        raise ValueError("x and y doesn't have same lenght")
    u = np.zeros(npts)
    v = np.zeros(npts)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, s],
                  [-s, c]])
    for i in range(npts):
        u[i], v[i] = R @ np.array([x[i], y[i]])
    return u, v

def fit_ellipse(x: Iterable[float], y: Iterable[float]):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def main():
    alldatavalues = read_all_files()
    for folder, datafolder in alldatavalues.items():
        plt.figure()
        phases = []
        for frequency, data in datafolder.items():
            xvalues = data[:, 1]  # Acceleration
            yvalues = data[:, 2]  # Force
            coefs = fit_ellipse(xvalues, yvalues)
            x0, y0, ap, bp, e, phi = cart_to_pol(coefs)
            phases.append(phi)
            theta = np.linspace(0, 2*np.pi, 129)
            xelipse = ap*np.cos(theta)
            yelipse = bp*np.sin(theta)
            u, v = rotate_clockwise(xelipse, yelipse, -phi)
            # plt.plot(xvalues, yvalues, ls="dotted", label="ori"+str(frequency))
            plt.plot(u+x0, v+y0, label="est" + str(frequency))
        phases = np.array(phases) - np.pi
        maxdevi = np.max(np.abs(phases - np.mean(phases)))/2
        for i, ph in enumerate(phases):
            if np.abs(ph-np.mean(phases)) > 0.99*maxdevi:
                phases[i+1:] += np.pi
                break
        plt.legend()
        plt.figure()
        plt.scatter(datafolder.keys(), phases)
        plt.xlabel(r"Frequency $f$ (Hz)")
        plt.ylabel(r"Phase $\phi$ (Hz)")
        plt.title(folder)
        plt.axhline(y=0, ls="dashed")
        plt.axhline(y=np.pi/2, ls="dashed")
        plt.axhline(y=-np.pi/2, ls="dashed")
        plt.axhline(y=np.pi, ls="dashed")
        plt.axhline(y=-np.pi, ls="dashed")
        plt.show()

if __name__ == "__main__":
    main()
    