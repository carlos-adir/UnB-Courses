import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple

def plot_field(xmesh: Tuple[float], ymesh: Tuple[float], zvals: np.ndarray, ax = None, contour=True):
    xmesh = np.array(xmesh, dtype="float64")
    ymesh = np.array(ymesh, dtype="float64")
    zvals = np.array(zvals, dtype="float64")
    assert xmesh.ndim == 1
    assert ymesh.ndim == 1
    assert zvals.ndim == 2
    assert zvals.shape == (len(xmesh), len(ymesh))
    x, y = np.meshgrid(xmesh, ymesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()
    dx = 0.05*(xmesh[-1]-xmesh[0])/2.
    dy = 0.05*(ymesh[-1]-ymesh[0])/2.
    extent = [xmesh[0]-dx, xmesh[-1]+dx, ymesh[0]-dy, ymesh[-1]+dy]
    im = ax.imshow((zvals.T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    if contour:
        cp = ax.contour(x, y, zvals.T, 10, colors="k")
    div  = make_axes_locatable(ax)
    cax  = div.append_axes('bottom', size='5%', pad=0.6)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    return ax


def plot_all_fields(xmesh, ymesh, S, U, V, W, P, title: str=None):

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    plot_field(xmesh, ymesh, S, contour=True, ax=axes[0])
    plot_field(xmesh, ymesh, U, contour=True, ax=axes[1])
    plot_field(xmesh, ymesh, V, contour=True, ax=axes[2])
    plot_field(xmesh, ymesh, W, contour=True, ax=axes[3])
    plot_field(xmesh, ymesh, P, contour=True, ax=axes[4])
    axes[0].set_title(r"Linha de corrente $S$")
    axes[1].set_title(r"Horizontal speed $u$")
    axes[2].set_title(r"Vertical speed $v$")
    axes[3].set_title(r"Vorticidade $W$")
    axes[4].set_title(r"Pressure $p$")
    if title is not None:
        fig.suptitle(title)
    for i in range(5):
        # axes[i].set_xlabel(r"Dimensao $x$")
        # axes[i].set_ylabel(r"Dimensao $y$")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        # [axes[i].axvline(x=xi, ls="dotted", color="k") for xi in list(set(Nx.knotvector))[1:-1]]
        # [axes[i].axhline(y=yj, ls="dotted", color="k") for yj in list(set(Ny.knotvector))[1:-1]]



def plot_animated_field(tmesh: Tuple[float], xmesh: Tuple[float], ymesh: Tuple[float], U: np.ndarray, V: np.ndarray, P: np.ndarray, ax = None, contour=True, repeat=True, timecycle = 10 ):
    tmesh = np.array(tmesh, dtype="float64")
    xmesh = np.array(xmesh, dtype="float64")
    ymesh = np.array(ymesh, dtype="float64")
    U = np.array(U, dtype="float64")
    V = np.array(V, dtype="float64")
    P = np.array(P, dtype="float64")
    assert tmesh.ndim == 1
    assert xmesh.ndim == 1
    assert ymesh.ndim == 1
    assert U.shape == (len(tmesh), len(xmesh), len(ymesh))
    assert V.shape == (len(tmesh), len(xmesh), len(ymesh))
    assert P.shape == (len(tmesh), len(xmesh), len(ymesh))

    x, y = np.meshgrid(xmesh, ymesh)
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig = plt.gcf()
    dx = 0.05*(xmesh[-1]-xmesh[0])/2.
    dy = 0.05*(ymesh[-1]-ymesh[0])/2.
    extent = [xmesh[0]-dx, xmesh[-1]+dx, ymesh[0]-dy, ymesh[-1]+dy]
    imU = ax[0].imshow((U[0].T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    imV = ax[1].imshow((V[1].T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    imP = ax[2].imshow((P[2].T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
   
    cp = []
    for i, F in enumerate([U, V, P]):
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        if contour:
            cp.append(ax[i].contour(x, y, F[0].T, 10, vmin=np.min(F), vmax=np.min(F), colors="k"))
    divU  = make_axes_locatable(ax[0])
    caxU  = divU.append_axes('bottom', size='5%', pad=0.6)
    cbarU = plt.colorbar(imU, cax=caxU, orientation='horizontal')
    divV  = make_axes_locatable(ax[1])
    caxV  = divV.append_axes('bottom', size='5%', pad=0.6)
    cbarV = plt.colorbar(imV, cax=caxV, orientation='horizontal')
    divP  = make_axes_locatable(ax[2])
    caxP  = divP.append_axes('bottom', size='5%', pad=0.6)
    cbarP = plt.colorbar(imP, cax=caxP, orientation='horizontal')
    # line1 = ax.axvline(x=0.5, ls="dotted", color="k")
    # line2 = ax.axhline(y=0.5, ls="dotted", color="k")

    def init():
        imU.set_data(U[0].T[::-1])
        imV.set_data(V[0].T[::-1])
        imP.set_data(P[0].T[::-1])
        if contour:
            return [imU, imV, imP] + cp[0].collections + cp[1].collections + cp[2].collections
        return [imU, imV, imP]

    # animation function.  This is called sequentially
    def animate(i):
        imU.set_data(U[i].T[::-1])
        imV.set_data(V[i].T[::-1])
        imP.set_data(P[i].T[::-1])
        if contour:
            cp[0] = ax[0].contour(x, y, U[i].T, 10, vmin=np.min(U), vmax=np.min(U), colors="k")
            cp[1] = ax[1].contour(x, y, V[i].T, 10, vmin=np.min(V), vmax=np.min(V), colors="k")
            cp[2] = ax[2].contour(x, y, P[i].T, 10, vmin=np.min(P), vmax=np.min(P), colors="k")
            return [imU, imV, imP] + cp[0].collections+ cp[1].collections + cp[2].collections
        return [imU, imV, imP]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(tmesh), interval=timecycle, blit=True)
    return anim