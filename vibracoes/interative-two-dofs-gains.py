import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
 
# Create a subplot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)

initvals = [0.3, 0.0, 0.2, 0.3]  # xi1, rm, rw, xi2

def fill_X(M, C, K, wvals):
    for i, w in enumerate(wvals):
        Xg[:, i] = np.abs(np.linalg.lstsq(K+1j*w*C-w**2*M, [1, 0], rcond=-1)[0])
    return Xg

def transform_params(xi1, rm, rw, xi2):
    m1 = 1
    k1 = 1
    wn1 = np.sqrt(k1/m1)
    c1 = 2*xi1*np.sqrt(k1*m1)
    m2 = m1*rm
    wn2 = wn1*rw
    k2 = m2*(wn2**2)
    c2 = 2*xi2*np.sqrt(k2*m2)
    return [m1, c1, k1], [m2, c2, k2]
def get_matrix(sys1, sys2):
    m1, c1, k1 = sys1
    m2, c2, k2 = sys2
    M = np.array([[m1, 0], [0, m2]])
    C = np.array([[c1+c2, -c2], [-c2, c2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    return M, C, K

def find_w_minimalX1g(sys1, sys2):
    m1, c1, k1 = sys1
    m2, c2, k2 = sys2
    wops = np.linspace(0, 3, 65)
    w = sp.symbols("w", real=True, positive=True)
    k2m2w2 = k2-m2*w**2
    func = (2*m1*k2m2w2+c1*c2)*(c2**2 * w**2 + k2m2w2**2)
    func += m2*(c2**2 * w**2 * (2*k2-m2*w**2)+2*k2**2 * k2m2w2)
    dfunc = sp.diff(func, w)
    func = sp.lambdify(w, func)
    dfunc = sp.lambdify(w, dfunc)
    for k in range(3):
        for j, wop in enumerate(wops):
            for i in range(15):
                dfwop = dfunc(wop)
                if np.abs(dfwop) < 1e-9:
                    wop = -1
                    break
                wop -= func(wop)/dfwop
            wops[j] = wop
        wfinal = []
        for w in wops:
            if w > 3:
                continue
            if w < 0:
                continue
            if len(wfinal) == 0:
                wfinal.append(w)
                continue
            if np.any(np.abs(np.array(wfinal)-w)<1e-9):
                continue
            wfinal.append(w)
    return wfinal

 
# Create 3 axes for 3 sliders red,green and blue
axxi1 = plt.axes([0.25, 0.25, 0.65, 0.03])
axrm = plt.axes([0.25, 0.2, 0.65, 0.03])
axrw = plt.axes([0.25, 0.15, 0.65, 0.03])
axxi2 = plt.axes([0.25, 0.1, 0.65, 0.03])
 
# Create a slider from 0.0 to 1.0 in axes axred
# with 0.6 as initial value.
slidxi1 = Slider(axxi1, 'xi1', 0.01, 1.0, initvals[0])
slidrm = Slider(axrm, 'm2/m1', 0.0, 3.0, initvals[1])
slidrw = Slider(axrw, 'wn2/wn1', 0.0, 3.0, initvals[2])
slidxi2 = Slider(axxi2, 'xi2', 0.01, 1.0, initvals[3])
 
# Create function to be called when slider value is changed

wplot = np.linspace(0.01, 3, 1025)
Xg = np.zeros((2, len(wplot)), dtype="float64")
sys1, sys2 = transform_params(*initvals)
M, C, K = get_matrix(sys1, sys2)
Xg = fill_X(M, C, K, wplot)
ax.plot(wplot, Xg[0], label=r"$X_1$")
ax.plot(wplot, Xg[1], label=r"$X_2$")
ax.axvline(x=1, color="b", ls="dashed", label=r"$\omega_{n1}$")
if sys2[0]:
    wops = find_w_minimalX1g(sys1, sys2)
    wn2 = np.sqrt(sys2[2]/sys2[0])
    ax.axvline(x=wn2, color="r", ls="dashed", label=r"$\omega_{n2}$")
    print("wops = ", wops)
    for wop in wops:
        ax.axvline(x=wop, color="g", ls="dashed")
ax.legend()
ax.grid()
ax.set_xlabel(r"Relative frequency $r=\omega/\omega_{n1}$")
ax.set_ylabel(r"Gain of functions")


def update(val):
    xi1 = slidxi1.val
    rm = slidrm.val
    rw = slidrw.val
    xi2 = slidxi2.val
    sys1, sys2 = transform_params(xi1, rm, rw, xi2)
    M, C, K = get_matrix(sys1, sys2)
    Xg = fill_X(M, C, K, wplot)
    wop = find_w_minimalX1g(sys1, sys2)
    ax.clear()
    ax.axvline(x=1, color="b", ls="dashed", label=r"$\omega_{n1}$")
    ax.plot(wplot, Xg[0], label=r"$X_1$")
    ax.plot(wplot, Xg[1], label=r"$X_2$")
    if sys2[0]:
        wops = find_w_minimalX1g(sys1, sys2)
        wn2 = np.sqrt(sys2[2]/sys2[0])
        ax.axvline(x=wn2, color="r", ls="dashed", label=r"$\omega_{n2}$")
        print("wops = ", wops)
        for wop in wops:
            ax.axvline(x=wop, color="g", ls="dashed")
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"Relative frequency $r=\omega/\omega_{n1}$")
    ax.set_ylabel(r"Gain of functions")

# Call update function when slider value is changed
slidxi1.on_changed(update)
slidrm.on_changed(update)
slidrw.on_changed(update)
slidxi2.on_changed(update)

# Display graph
plt.show()