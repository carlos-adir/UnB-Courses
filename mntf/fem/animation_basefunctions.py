import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from compmec import nurbs
from helper import getD

ny, py = 101, 2
Uy = nurbs.GeneratorKnotVector.uniform(py, ny)
Ny = nurbs.SplineBaseFunction(Uy)
nframes = 1000
xsample = np.linspace(0, 1, 129)
ysample = np.linspace(0, 1, nframes)

if False:
    basefunctions = [lambda x: np.cos(np.pi*x),
                    lambda x: np.sin(np.pi*x),
                    lambda x: np.exp(x),
                    lambda x: np.ones(len(x)),
                    lambda x: x,
                    lambda x: x**2]
    derivadas = [lambda x: -np.pi*np.sin(np.pi*x),
                 lambda x: np.pi*np.cos(np.pi*x),
                 lambda x: np.exp(x),
                 lambda x: np.zeros(len(x)),
                 lambda x: np.ones(len(x)),
                 lambda x: 2*x]
    nomes = [r"$\cos \pi x$", r"$\sin \pi x$", r"$\exp x$", r"$1$", r"$x$", r"$x^2$"]
    nx = len(basefunctions)
else:
    nx, px = 7, 2
    Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
    Nx = nurbs.SplineBaseFunction(Ux)
    basefunctions = [Nx[i] for i in range(nx)]
    dNx = nurbs.SplineBaseFunction(Ux)
    dNx.derivate()
    derivadas = [dNx[i] for i in range(nx)]
    nomes = [r"$N_{%d,%d}$"%(i,px) for i in range(nx)]
Lx = np.zeros((nx, len(xsample)))
dLx = np.zeros((nx, len(xsample)))
for i, func in enumerate(basefunctions):
    Lx[i, :] = func(xsample)
for i, dfunc in enumerate(derivadas):
    dLx[i, :] = dfunc(xsample)
Ly = Ny(ysample)
ctrlpoints = 2*np.random.rand(nx, ny)-1
ctrlpoints[:, -1] = ctrlpoints[:, 0]  # Continuidade em y = 0
Dpy = np.eye(ny)
for j in range(1, py):  # Continuidade das derivadas
    Dpy = Dpy @ getD(py, Uy)
    R = Dpy @ (Ny[:,py-j](1) - Ny[:,py-j](0))
    ctrlpoints[:, -j-1] = -(ctrlpoints[:, :-j-1] @ R[:-j-1] + ctrlpoints[:,-j:] @ R[-j:])/R[-j-1]

# index = 2
# ctrlpoints[index+1:, :] = 0
# ctrlpoints[:index, :] = 0

weights = Lx.T @ ctrlpoints
allpoints =  Lx.T @ ctrlpoints @ Ly
dweights = dLx.T @ ctrlpoints
dallpoints =  dLx.T @ ctrlpoints @ Ly

fig, axis = plt.subplots(1, 2, figsize=(12, 4))
maxy =  np.max(allpoints)
miny = np.min(allpoints)
dy = maxy-miny
maxdy =  np.max(dallpoints)
mindy = np.min(dallpoints)
ddy = maxdy-mindy
axis[0].set_xlim(0, 1)
axis[0].set_ylim(miny-0.05*dy, maxy+0.05*dy)
axis[1].set_xlim(0, 1)
axis[1].set_ylim(mindy-0.05*ddy, maxdy+0.05*ddy)
axis[0].set_title("Valor da funcao")
axis[1].set_title("Valor da derivada")
mainline, = axis[0].plot([], [], lw = 3)
mainline2, = axis[1].plot([], [], lw = 3)
lines = []
lines2 = []
for i in range(nx):
    line, = axis[0].plot([], [], ls="dashed", label=nomes[i])
    line2, = axis[1].plot([], [], ls="dashed", label=f"{nomes[i]}'")
    lines.append(line)
    lines2.append(line2)
if isinstance(basefunctions[0], nurbs.basefunctions.SplineEvaluatorClass):
    [axis[0].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Ux))]
    [axis[1].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Ux))]
axis[0].grid()
axis[0].legend()
axis[1].grid()
axis[1].legend()
def animate(i):
    mainline.set_data(xsample, allpoints[:, i])
    mainline2.set_data(xsample, dallpoints[:, i])
    for k, line in enumerate(lines):
        line.set_data(xsample, Lx[k] * (ctrlpoints[k] @ Ly[:, i]))
    for k, line2 in enumerate(lines2):
        line2.set_data(xsample, dLx[k] * (ctrlpoints[k] @ Ly[:, i]))
    return [mainline, mainline2] + lines + lines2

anim = animation.FuncAnimation(fig, animate, 
                    frames = nframes, interval = 20, blit = True)
plt.show()
# anim.save('randomPathBasefunctions.mp4', writer = 'ffmpeg', fps = 60)