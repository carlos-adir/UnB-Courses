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
frequency = data[:, 0]
magnitude = data[:, 1]
phase = data[:, 2]
# intervalofrequencia = [15, 70]
intervalofrequencia = [min(frequency), max(frequency)]
mask = (frequency > intervalofrequencia[0]) * (frequency < intervalofrequencia[1])
freq = frequency[mask]
mag = magnitude[mask]
phi = phase[mask]
G = mag*np.exp(1j*phi*np.pi/180)

# k, c, m, w = sp.symbols("k c m w", real=True, positive=True)
# G = -w**2/(k-w**2*m + 1j*w*c)
# absG = sp.Abs(G)
# diffe = sp.diff(absG, w)
# diffe = sp.simplify(diffe)
# print("equation = ")
# print(diffe)
# solution = sp.solve(diffe, w, )
# print("solution = ", solution)
# wabsGmax = solution[-1]
# print("wabsGmax = ")
# print(wabsGmax)


wope = 28  # Hz
m1 = 73  # kg


xi1 = 0.99996
wn1 = 80  # Hz
c1 = 2*xi1*wn1
k1 = m1*wn1**2
angle = 0.85*(np.pi/2)  # To fit interpolation
uvals = np.linspace(0, angle, 1025)
rplot1 = np.exp(-np.tan(uvals)*np.log(wn1/min(intervalofrequencia))/np.tan(angle))
rplot2 = np.exp(np.tan(uvals)*np.log(max(intervalofrequencia)/wn1)/np.tan(angle))
wplot = wn1 * np.concatenate([np.flipud(rplot1), rplot2])
# wabsGwmax = np.sqrt(-(c1**2)/2 + k1*m1)/m1
wabsGwmax = k1*np.sqrt(2/(2*k1*m1-c1**2))
wplot = np.array(wplot.tolist() + [wabsGwmax])
wplot.sort()
# Gw = 1/(k1+1j*c1*wplot-m1*wplot**2)
Gw = -wplot**2/(k1+1j*c1*wplot-m1*wplot**2)

F0 = np.max(magnitude)/np.max(np.abs(Gw))
F0 = 100
gain = Gw * F0 # ddot(x) = G(w) * F




fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
xi = 0.6
wn = 0.2
F0 = 0.5

# Create 3 axes for 3 sliders red,green and blue
axxi = plt.axes([0.25, 0.2, 0.65, 0.03])
axwn = plt.axes([0.25, 0.15, 0.65, 0.03])
axF0 = plt.axes([0.25, 0.1, 0.65, 0.03])
 
# Create a slider from 0.0 to 1.0 in axes axred
# with 0.6 as initial value.
xislider = Slider(axxi, r'$\xi$', 0.0, 1.0, 0.6)
 
# Create a slider from 0.0 to 1.0 in axes axgreen
# with 0.2 as initial value.
wnslider = Slider(axwn, r'$\omega_n$', 10, 100, 75)
 
# Create a slider from 0.0 to 1.0 in axes axblue
# with 0.5(default) as initial value
F0slider = Slider(axF0, r'$F_0$', 0.0, 100, 0)
 
# Create function to be called when slider value is changed

def update(val):
    xi = xislider.val
    wn = wnslider.val
    F0 = F0slider.val
    rplot1 = np.exp(-np.tan(uvals)*np.log(wn1/min(intervalofrequencia))/np.tan(angle))
    rplot2 = np.exp(np.tan(uvals)*np.log(max(intervalofrequencia)/wn1)/np.tan(angle))
    wplot = wn1 * np.concatenate([np.flipud(rplot1), rplot2])
    c = 2*m1*xi*wn
    k = m1*wn**2
    Gw = wplot**2/(k + 1j*c*wplot - m1*wplot**2)
    xddot = F0*Gw
    ax.cla()
    if False:  # Ganho
        ax.scatter(freq, np.abs(G), label="sample", marker=".", color="tab:blue")
        ax.plot(wplot, np.abs(xddot), label=r"$\dfrac{-\omega^2F_0}{k+ic\omega-m\omega^2}$", color="r", ls="dotted")
        ax.axvline(x=wope, color="g", ls="dashed")
        ax.set_xlabel(r"Frequência $\omega$ (Hz)")
        ax.set_ylabel(r"Ganho")
    elif False:  # Fase
        ax.scatter(freq, (180/np.pi)*np.angle(G), label="sample", marker=".", color="tab:blue")
        ax.plot(wplot, (180/np.pi)*np.angle(xddot), label=r"$\dfrac{\omega^2F_0}{k+ic\omega-m\omega^2}$", color="r", ls="dotted")
        ax.axvline(x=wope, color="g", ls="dashed")
        ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        ax.set_xlabel(r"Frequência $\omega$ (Hz)")
        ax.set_ylabel(r"Fase (graus)")
        ax.set_ylim([-200, 200])
    else:
        ax.plot(np.real(G), np.imag(G), ls="dotted", marker=".", label="sample")
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
F0slider.on_changed(update)
 
# Create axes for reset button and create button
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color='gold',
                hovercolor='skyblue')
 
# Create a function resetSlider to set slider to
# initial values when Reset button is clicked
 
def resetSlider(event):
    xislider.reset()
    wnslider.reset()
    F0slider.reset()
 
# Call resetSlider function when clicked on reset button
button.on_clicked(resetSlider)
 
# Display graph
plt.show()





# plt.figure()
# plt.plot(freq, np.abs(G), label="sample")
# plt.plot(wplot, np.abs(gain), label="esti")
# plt.xlabel(r"Frequência $\omega$ (Hz)")
# plt.ylabel(r"Magnitude ganho $|G(\omega)|$")
# plt.axvline(x=wope, ls="dashed", color="g")
# plt.grid()

# plt.figure()
# plt.plot(freq, (180/np.pi)*np.angle(G), label="sample")
# plt.plot(wplot, -180+(180/np.pi)*np.angle(gain), label="esti")
# plt.xlabel(r"Frequência $\omega$ (Hz)")
# plt.ylabel(r"Fase $arg G(\omega)$")
# plt.gca().set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
# plt.axvline(x=wope, ls="dashed", color="g")
# plt.grid()

# plt.figure()

# plt.show()


