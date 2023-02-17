import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from compmec import nurbs
from helper import getD
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple
from femnavierhelper import *

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


def plot_all_fields(Nx: nurbs.SplineBaseFunction, Ny: nurbs.SplineBaseFunction, S, title: str=None):
    xplot = np.linspace(0, 1, 1025)
    yplot = np.linspace(0, 1, 1025)
    px, nx = Nx.degree, Nx.npts
    py, ny = Ny.degree, Ny.npts
    Dpx = getD(px, Nx.knotvector)
    Dpx1 = getD(px-1, Nx.knotvector)
    Dpy = getD(py, Ny.knotvector)
    Dpy1 = getD(py-1, Ny.knotvector)
    Lx = Nx[:, px](xplot)
    dLx = Dpx @ Nx[:, px-1](xplot)
    ddLx = Dpx @ Dpx1 @ Nx[:, px-2](xplot)
    Ly = Ny[:, py](yplot)
    dLy = Dpy @ Ny[:, py-1](yplot)
    ddLy = Dpy @ Dpy1 @ Ny[:, py-2](yplot)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    splot = Lx.T @ S @ Ly
    plot_field(xplot, yplot, splot, contour=True, ax=axes[0])
    uplot = Lx.T @ S @ dLy  # U
    plot_field(xplot, yplot, uplot, contour=True, ax=axes[1])
    vplot = dLx.T @ S @ dLy  # V
    plot_field(xplot, yplot, vplot, contour=True, ax=axes[2])
    wplot = -(ddLx.T @ S @ Ly + Lx.T @ S @ ddLy)  # W
    plot_field(xplot, yplot, wplot, contour=True, ax=axes[3])
    P = compute_pressure_from_current_line(Nx, Ny, S)
    zplot = Lx.T @ P @ Ly
    plot_field(xplot, yplot, zplot, contour=True, ax=axes[4])
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



def plot_animated_field(tmesh: Tuple[float], xmesh: Tuple[float], ymesh: Tuple[float], zvals: np.ndarray, ax = None, contour=True, repeat=True, timecycle = 10 ):
    tmesh = np.array(tmesh, dtype="float64")
    xmesh = np.array(xmesh, dtype="float64")
    ymesh = np.array(ymesh, dtype="float64")
    zvals = np.array(zvals, dtype="float64")
    assert tmesh.ndim == 1
    assert xmesh.ndim == 1
    assert ymesh.ndim == 1
    assert zvals.ndim == 3
    assert zvals.shape == (len(tmesh), len(xmesh), len(ymesh))

    x, y = np.meshgrid(xmesh, ymesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()
    dx = 0.05*(xmesh[-1]-xmesh[0])/2.
    dy = 0.05*(ymesh[-1]-ymesh[0])/2.
    extent = [xmesh[0]-dx, xmesh[-1]+dx, ymesh[0]-dy, ymesh[-1]+dy]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    zmin, zmax = np.min(zvals), np.max(zvals)
    im = ax.imshow((zvals[0].T)[::-1], cmap="viridis", interpolation='nearest', aspect='auto', extent=extent)
    if contour:
        cp = [ax.contour(x, y, zvals[0].T, 10, vmin=zmin, vmax=zmax, colors="k")]
    div  = make_axes_locatable(ax)
    cax  = div.append_axes('bottom', size='5%', pad=0.6)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    line1 = ax.axvline(x=0.5, ls="dotted", color="k")
    line2 = ax.axhline(y=0.5, ls="dotted", color="k")

    def init():
        im.set_data(zvals[0].T[::-1])
        if contour:
            return [im, line1, line2] + cp[0].collections
        return [im, line1, line2]

    # animation function.  This is called sequentially
    def animate(i):
        im.set_array(zvals[i].T[::-1])
        if contour:
            cp[0] = ax.contour(x, y, zvals[i].T, 10, vmin=zmin, vmax=zmax, colors="k")
            return [im, line1, line2] + cp[0].collections
        return [im, line1, line2]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=zvals.shape[0], interval=timecycle, blit=True)
    return anim