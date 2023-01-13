import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from compmec import nurbs
from helper import getD

ny, py = 44, 1
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
    nomes = [r"$\cos \pi x$", r"$\sin \pi x$", r"$\exp x$", r"$1$", r"$x$", r"$x^2$"]
    nx = len(basefunctions)
else:
    nx, px = 4, 1
    Ux = nurbs.GeneratorKnotVector.uniform(px, nx)
    Nx = nurbs.SplineBaseFunction(Ux)
    basefunctions = [Nx[i] for i in range(nx)]
    nomes = [r"$N_{%d,%d}$"%(i,px) for i in range(nx)]
Lx = np.zeros((nx, len(xsample)))
for i, func in enumerate(basefunctions):
    Lx[i, :] = func(xsample)
Ly = Ny(ysample)
ctrlpoints = 2*np.random.rand(nx, ny)-1
ctrlpoints[:, -1] = ctrlpoints[:, 0]  # Continuidade em y = 0
Dpy = np.eye(ny)
for j in range(1, py):  # Continuidade
    Dpy = Dpy @ getD(py, Uy)
    R = Dpy @ (Ny[:,py-j](1) - Ny[:,py-j](0))
    ctrlpoints[:, -j-1] = -(ctrlpoints[:, :-j-1] @ R[:-j-1] + ctrlpoints[:,-j:] @ R[-j:])/R[-j-1]

# index = 5
# ctrlpoints[index+1:, :] = 0
# ctrlpoints[:index, :] = 0

weights = Lx.T @ ctrlpoints
allpoints =  Lx.T @ ctrlpoints @ Ly

fig = plt.figure()
maxy =  np.max(allpoints)
miny = np.min(allpoints)
dy = maxy-miny
axis = plt.axes(xlim=(0, 1), ylim=(miny-0.1*dy, maxy+0.1*dy))
mainline, = axis.plot([], [], lw = 3)
lines = []
for i in range(nx):
    line, = axis.plot([], [], ls="dashed", label=nomes[i])
    lines.append(line)
if isinstance(basefunctions[0], nurbs.basefunctions.SplineEvaluatorClass):
    [axis.axvline(x=xi, ls="dotted", color="k") for xi in list(set(Ux))]
axis.grid()
axis.legend()
def animate(i):
    for k, line in enumerate(lines):
        line.set_data(xsample, Lx[k] * (ctrlpoints[k] @ Ly[:, i]))
    mainline.set_data(xsample, allpoints[:, i])
    return [mainline] + lines 

anim = animation.FuncAnimation(fig, animate, 
                    frames = nframes, interval = 20, blit = True)
plt.show()
# anim.save('randomPathBasefunctions.mp4', writer = 'ffmpeg', fps = 60)