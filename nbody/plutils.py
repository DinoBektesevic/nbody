import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from scipy.stats import maxwell

__all__ = ["plot_Etot",
           "plot_positions", "plot_all_positions",
           "hist_velocities", "hist_all_velocities",  "plot_avg_v",
           "snapshot_data", "sim_data"]


############################################################
###################       Data IO        ###################
############################################################
def snapshot_data(filepath):
    return np.genfromtxt(filepath, names=True)

def sim_data(folderpath):
    ls = os.listdir(folderpath)
    ls.sort()
    data = []
    for filename in ls:
        if ("snapshot" in filename) and ("png" not in filename):
            data.append(snapshot_data(join(folderpath, filename)))

    return np.asarray(data)




############################################################
##################        Plotting functions        ########
############################################################
def unwrap_kwargs(f):
    """
    Unpack keyword arguments and send them in as positional keyword args to the
    decorated function to create a uniform interface to the matplotlib
    functionality.
    """
    def callf(*args, **kwargs):
        show  = True     if kwargs.get("show")   is None else kwargs.pop("show")
        s     = 100      if kwargs.get("s")      is None else kwargs.pop("s")
        title = None     if kwargs.get("title")  is None else kwargs.pop("title")
        bins  = 50       if kwargs.get("bins")   is None else kwargs.pop("bins")
        xlim  = (-2, 2)  if kwargs.get("xlim")   is None else kwargs.pop("xlim")
        ylim  = (0, 600) if kwargs.get("ylim")   is None else kwargs.pop("ylim")
        label = None     if kwargs.get("label")  is None else kwargs.pop("label")
        fname = None     if kwargs.get("fname")  is None else kwargs.pop("fname")
        xlabel= None     if kwargs.get("xlabel") is None else kwargs.pop("xlabel")
        ylabel= None     if kwargs.get("ylabel") is None else kwargs.pop("ylabel")
        f(*args, s=s, bins=bins, xlim=xlim, ylim=ylim, label=label, show=show,
          title=title, fname=fname, xlabel=xlabel, ylabel=ylabel, **kwargs)
    return callf



@unwrap_kwargs
def plot_positions(ax, x, y, s, label, show, title, fname, xlabel, ylabel, **kwargs):
    """
    Scatter plots x and y coordinates on axis ax. 
    """
    if fname is None and not show:
        raise ValueError("Please provide the name of the file image will be saved to.")

    out = ax.scatter(x, y, s=s)

    if xlabel is not None: ax.set_xlabel("x")
    if ylabel is not None: ax.set_ylabel("y")
    if title is not None:  ax.set_title(title)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    ax.set_aspect('equal', 'box')

    if show:
        return ax
    else:
        plt.grid(True)
        plt.tight_layout()
        sname = "snapshot_{0:0<14}_pos".format(float(fname))
        sname = sname.replace(".", "_") + ".png"
        plt.savefig(sname)
        plt.clf()
        plt.close("all")

def plot_all_positions(folderpath=".", **kwargs):
    """
    Calls plot_positions for every snapshot found in a given folder.
    """
    ls = os.listdir(folderpath)
    ls.sort()
    for filename in ls:
        if ("snapshot" in filename) and ("png" not in filename):
            fig, ax = plt.subplots(figsize=figsize)
            d = nbody.snapshot_data(filename)
            plot_positions(ax, d["x"], d["y"], **kwargs)




@unwrap_kwargs
def hist_velocities(ax, v, bins, xlim, ylim, label, fname, title, show, s,
                    xlabel, ylabel, **kwargs):
    """
    Given an axis ax and velocities v plots a historam of v.
    """

    if fname is None and not show:
        raise ValueError("Please provide the name of the file image will be saved to.")

    out = ax.hist(v, label=label, bins=bins, **kwargs)

    if xlabel is None: ax.set_xlabel(r"$v [1/v_s]$")
    else: ax.set_ylabel(xlabel)
    if ylabel is None: ax.set_ylabel("N [counts]")
    else: ax.set_ylabel(ylabel)

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if show:
        return ax
    else:
        plt.grid(True)
        plt.tight_layout()
        sname = "snapshot_{0:0<14}_vel".format(float(fname))
        sname = sname.replace(".", "_") + ".png"
        plt.savefig(sname)
        plt.clf()
        plt.close("all")

def hist_all_velocities(folderpath=".", **kwargs):
    """
    Calls hist_velocities for every snapshot found in a given folder.
    """
    if fname is None and not show:
        raise ValueError("Please provide the name of the file image will be saved to.")

    ls = os.listdir(folderpath)
    ls.sort()
    for filename in ls:
        if ("snapshot" in filename) and ("png" not in filename):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
            d = nbody.snapshot_data(filename)
            hist_velocities(axes[0], d["vx"], **kwargs)
            hist_velocities(axes[0], d["vy"], **kwargs)

def plot_avg_v(ax, data, bins=100, which=-1, extent=(-2, 2), **kwargs):
    """
    Given a list of snapshots 'data', plots a normalized histogram of the average
    magnitude of the particle velocity v = sqrt(vx**2 + vy**2).
    """

    vxavgc = np.zeros((bins, ))
    vyavgc = np.zeros((bins, ))
    vavgc = np.zeros((bins, ))

    vxs, vys, vs = [], [], []
    for d in data:
        vx = d[which]["vx"]
        vy = d[which]["vy"]
        v = np.sqrt(vx**2 + vy**2)
        vxs.extend(vx)
        vys.extend(vy)
        vs.extend(v)

    vhc,  binedg = np.histogram(vs, bins=bins, range=(extent[0], extent[-1]),
                                density=True)
    w=binedg[1]-binedg[0]
    ax.bar(binedg[:-1], vhc, label="Sim.", width=w, alpha=0.3)

    ax.set_ylabel("N [counts]")
    ax.set_xlabel(r"$v [1/v_s]$")

    ax.legend()
    return ax




def plot_Etot(ax, data, doprint=True, **kwargs):
    """
    Calculates the total energy of a snapshot, assumed to be the sum of all
    individual kinetic particle energies, and plots it as a function of time. 
    """
    t = [d["t"][0] for d in data] 
    ekin = [sum( d["m"]*(d["vx"]**2+d["vy"]**2) ) for d in data]

    ax.plot(t, np.mean(ekin)-ekin, label=r"$E_{\mathrm{kin}}$", **kwargs)

    ax.set_xlabel("t[s]")
    ax.set_ylabel("2*Ekin [kT]")
    ax.legend()
    if doprint:
        print("Average total energy of the system is {0}kT.".format(np.mean(ekin)/2.))
        print("Maximal total energy of the system is {0}kT.".format(np.max(ekin)/2.))
        print("Minimal total energy of the system is {0}kT.".format(np.min(ekin)/2.))
    return ax
