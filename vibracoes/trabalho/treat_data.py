import numpy as np
from matplotlib import pyplot as plt

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
intervalofrequencia = [15, 70]
mask = (frequency > intervalofrequencia[0]) * (frequency < intervalofrequencia[1])
freq = frequency[mask]
mag = magnitude[mask]
phi = phase[mask]
G = mag*np.exp(1j*phi*np.pi/180)

w = 28  # Hz
m2 = 1  # kg
k2 = m2 * (2*np.pi*w)**2
print("m2 = ", m2)
print("k2 = ", k2)



plt.figure()
plt.plot(freq, np.abs(G))
plt.xlabel(r"Frequência $\omega$ (Hz)")
plt.ylabel(r"Magnitude ganho $|G(\omega)|$")
plt.axvline(x=w, ls="dashed", color="g")
plt.grid()

plt.figure()
plt.plot(freq, (180/np.pi)*np.angle(G))
plt.xlabel(r"Frequência $\omega$ (Hz)")
plt.ylabel(r"Fase $arg G(\omega)$")
plt.gca().set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
plt.axvline(x=w, ls="dashed", color="g")
plt.grid()

plt.figure()
plt.plot(np.real(G), np.imag(G), ls="dotted", marker=".")
plt.xlabel(r"Parte real de $G(\omega)$")
plt.ylabel(r"Parte imaginaria de $G(\omega)$")
plt.grid()
plt.axis("equal")
plt.show()


