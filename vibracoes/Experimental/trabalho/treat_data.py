import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

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


pythonfolder = __file__.replace("treat_data.py", "").replace("\\", "/")
data = readlines(pythonfolder+"dados_manha/acc_frf_motor.txt", header=6)
pi2 = 2*np.pi
frequency = data[:, 0]*pi2
magnitude = data[:, 1]
phase = data[:, 2]
Greaded = magnitude*np.exp(1j*phase*np.pi/180)
# intervalofrequencia = np.array([15, 57])*pi2
intervalofrequencia = [min(frequency), max(frequency)]
print(intervalofrequencia)
mask = (frequency >= intervalofrequencia[0]) * (frequency <= intervalofrequencia[1])
# wsample = frequency[mask]
# Gsample = Greaded[mask]
wsample = frequency
Gsample = Greaded



def regressao(wi, Gsample, m):
    xi = np.real(Gsample)
    yi = np.imag(Gsample)
    Mat = np.zeros((3, 3))
    Vec = np.zeros(3)
    Mat[0, 0] = np.sum(wi**4)
    Mat[1, 0] = Mat[0, 1] = -np.sum(xi*wi**2)
    Mat[2, 0] = Mat[0, 2] = np.sum(yi*wi**3)
    Mat[1, 1] = np.sum(xi**2+yi**2)
    Mat[2, 2] = np.sum(wi**2 * (xi**2+yi**2))
    Vec[0] = -np.sum(m*xi*wi**4)
    Vec[1] = np.sum(m*wi**2*(xi**2+yi**2))
    F0, k, c = np.linalg.solve(Mat, Vec)
    return F0, k, c

wope = 28*pi2  # Hz
m1 = 13  # kg
angle = 0.85*(np.pi/2)  # To fit interpolation
uvals = np.linspace(0, angle, 1025)

if False:
    F0, k1, c1 = regressao(wsample[mask], Gsample[mask], m1)
    wn1 = np.sqrt(k1/m1)
    xi1 = c1/np.sqrt(4*k1*m1)
    rplot1 = np.exp(-np.tan(uvals)*np.log(wn1/min(wsample))/np.tan(angle))
    rplot2 = np.exp(np.tan(uvals)*np.log(max(wsample)/wn1)/np.tan(angle))
    wplot = wn1 * np.concatenate([np.flipud(rplot1), rplot2])
    mask = (wplot >= min(wsample)) * (wplot <= max(wsample))
    wplot = wplot[mask] 
    Gplot = F0*wplot**2/(k1+1j*c1*wplot-m1*wplot**2)
    print(" F0 = %.3f 1/N" % F0)
    print(" G0 = %.3f N" % (1/F0))
    print(" k1 = %.3f N/m" % k1)
    print(" c1 = %.3f N*s/m" % c1)
    print(" m1 = %.3f kg" % m1)
    print("wn1 = %.3f rad/s" % wn1)
    print("    = %.3f Hz" % (wn1/pi2))
    print("xi1 = %.3f" % xi1)

    plt.figure()
    plt.scatter(wsample/pi2, np.abs(Gsample), color="tab:blue", marker=".", label="sample")
    plt.plot(wplot/pi2, np.abs(Gplot), color="r", label="estimado")
    plt.xlabel(r"Frequência $\omega$ (Hz)")
    plt.ylabel(r"Magnitude ganho $|G(\omega)|$")
    plt.axvline(x=wope/pi2, ls="dashed", color="g")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.scatter(wsample/pi2, (180/np.pi)*np.angle(Gsample), marker=".", color="tab:blue", label="sample")
    plt.plot(wplot/pi2, (180/np.pi)*np.angle(Gplot), color="r", label="estimado")
    plt.xlabel(r"Frequência $\omega$ (Hz)")
    plt.ylabel(r"Fase $arg G(\omega)$")
    plt.gca().set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.axvline(x=wope/pi2, ls="dashed", color="g")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(np.real(Gsample), np.imag(Gsample), color="tab:blue", ls="dotted", marker=".", label="sample")
    plt.plot(np.real(Gplot), np.imag(Gplot), color="r", ls="dotted", label="estimado")
    plt.xlabel(r"Parte real de $G(\omega)$")
    plt.ylabel(r"Parte imaginaria de $G(\omega)$")
    plt.axis("equal")
    plt.grid()
    plt.legend()

    plt.show()

if True:
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    xi = 0.6
    wn = 0.2
    A = 0.5

    # Create 3 axes for 3 sliders red,green and blue
    axxi = plt.axes([0.25, 0.2, 0.65, 0.03])
    axwn = plt.axes([0.25, 0.15, 0.65, 0.03])
    axA = plt.axes([0.25, 0.1, 0.65, 0.03])
    xislider = Slider(axxi, r'$\xi$', 0.0, 1.0, 0.6)
    wnslider = Slider(axwn, r'$\omega_n$', 10, 100, 75)
    Aslider = Slider(axA, r'$A$', 0.0, 100, 0)
 
    # Create function to be called when slider value is changed

    def update(val):
        xi = xislider.val
        wn = pi2*wnslider.val
        A = Aslider.val
        rplot1 = np.exp(-np.tan(uvals)*np.log(wn/min(intervalofrequencia))/np.tan(angle))
        rplot2 = np.exp(np.tan(uvals)*np.log(max(intervalofrequencia)/wn)/np.tan(angle))
        wplot = wn * np.concatenate([np.flipud(rplot1), rplot2])
        mask = (wplot >= min(intervalofrequencia)) * (wplot <= max(intervalofrequencia))
        wplot = wplot[mask] 
        c = 2*m1*xi*wn
        k = m1*wn**2
        Gw = wplot**2/(k + 1j*c*wplot - m1*wplot**2)
        xddot = A*Gw
        ax.cla()
        if True:  # Ganho
            ax.scatter(wsample/pi2, np.abs(Gsample), label="sample", marker=".", color="tab:blue")
            ax.plot(wplot/pi2, np.abs(xddot), label=r"$\dfrac{-\omega^2A}{k+ic\omega-m\omega^2}$", color="r", ls="dotted")
            ax.axvline(x=wope, color="g", ls="dashed")
            ax.set_xlabel(r"Frequência $\omega$ (Hz)")
            ax.set_ylabel(r"Ganho")
        elif False:  # Fase
            ax.scatter(wsample/pi2, (180/np.pi)*np.angle(Gsample), label="sample", marker=".", color="tab:blue")
            ax.plot(wplot/pi2, (180/np.pi)*np.angle(xddot), label=r"$\dfrac{\omega^2F_0}{k+ic\omega-m\omega^2}$", color="r", ls="dotted")
            ax.axvline(x=wope, color="g", ls="dashed")
            ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
            ax.set_xlabel(r"Frequência $\omega$ (Hz)")
            ax.set_ylabel(r"Fase (graus)")
            ax.set_ylim([-200, 200])
        else:
            ax.plot(np.real(Gsample), np.imag(Gsample), ls="dotted", marker=".", label="sample")
            ax.plot(np.real(xddot), np.imag(xddot), ls="dotted", label="sample")
            ax.set_xlabel(r"Parte real de $G(\omega)$")
            ax.set_ylabel(r"Parte imaginaria de $G(\omega)$")
            ax.axis("equal")
        ax.grid()
        ax.legend()
        # ax.axvline(x=wn, color="r", ls="dashed")
 
    # Call update function when slider value is changed
    xislider.on_changed(update)
    wnslider.on_changed(update)
    Aslider.on_changed(update)
 
# Display graph
plt.show()





#

# plt.figure()

# plt.show()


