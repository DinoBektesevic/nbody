import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from scipy.stats import maxwell

__all__ = ["plot_Etot",
           "plot_positions", "plot_all_positions",
           "hist_velocities", "hist_all_velocities",
           "snapshot_data", "sim_data", "plot_avg_v"]


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
    Upack packed keyword arguments and send them in as positional keyword args
    to enable users to send in their own additionaseel kwargs to matplotlib funcs
    without them raising an error "unrecognized argument" but still letting you
    set the default values over several functions to avoid having long lines
    everywhere in the notebook.
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
    ls = os.listdir(folderpath)
    ls.sort()
    for filename in ls:
        if ("snapshot" in filename) and ("png" not in filename):
            fig, ax = plt.subplots(figsize=figsize)
            d = nbody.snapshot_data(filename)
            plot_positions(ax, d["x"], d["y"], **kwargs)
            #plot_positions(filename, show)

#ylabel= r"$N_{\mathrm{particles}}$" if kwargs.get("ylabel") is None else kwargs.pop("ylabel")



@unwrap_kwargs
def hist_velocities(ax, v, bins, xlim, ylim, label, fname, title, show, s,
                    xlabel, ylabel, **kwargs):
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



############################################################
##################        Plotting functions        ########
############################################################
def plot_avg_v(ax, data, bins=100, which=-1, extent=(-2, 2), **kwargs):

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

    vhc,  binedg = np.histogram(vs, bins=bins, range=(0, extent[-1]), density=True)
    w=binedg[1]-binedg[0]
    ax.bar(binedg[:-1], vhc, label="Sim.", width=w, alpha=0.3)

    ax.set_ylabel("N [counts]")
    ax.set_xlabel(r"$v [1/v_s]$")

    ax.legend()
    return ax



def plot_v(ax, data, N, skip=1, scale=0.1, **kwargs):
    vx, vy, v = averageN(data, N, skip)
    vweights = np.ones_like(v)/len(v)

    r = maxwell.rvs(scale=scale, size=10000)
    rweights = np.ones_like(r)/len(r)

    ax.hist(np.sqrt(v), weights=vweights, align="right", label="Sim.", **kwargs)
    ax.hist(r, weights=rweights, align="left", label="Max.-Boltz. dist.", **kwargs)

    ax.set_xlabel(r"$10^3[\frac{m}{s}]$")
    ax.set_ylabel(r"$N_\mathrm{particles}$")
    ax.legend()

    return ax




def plot_Etot(ax, data, doprint=True, **kwargs):
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
