# nbody

An event based N-body-in-a-box simulator for ASTRO507 class at University of Washington 2018. See the Jupyter Notebook for full details.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. 

### Prerequisites

Developed and executed using the following packages

```
python-3.6.2
numpy-1.13.1
matplotlib-2.0.2
```
but it will most likely run on other versions too.

### Installing

Clone this github repository and enter the target directory. Execute

```
$:> python
```
on the command prompt and import the package in the interpreter

```
>>> import nbody
```
It should work straight out of the box, if not - report a bug.

### Usage

To get a quick simulation working try
```
import nbody
s = nbody.TDMSystem(N_particles=10, t_total=1)
s.simulate()
```
which should simulate a 1 second system of 10 particles in a box. Output should be a series of files called `snapshot_t.ttt` which are just text files containing the simulation state at the time of the snapshot. Different intial conditions are availible as well.
```
s = TDMSystem(10, 1, pos_type="ordered", vel_type="identical")
s = TDMSystem(10, 1, pos_type="random", vel_type="ekin_constrained", p_kin=1, p_r=0.02)
s = TDMSystem(10, 1, pos_type="ordered", vel_type="identical", p_v=1, p_mass=2)
```
Reading the docstrings should be fairly self-explanatory. The initial state generation functions are availible from the `nbody` module. Additional keyword arguments are passed to the initial state generation function so specifying them at `TDMSystem` instatiation is possible as well.

```
s = TDMSystem(10, 1, pos_type="ordered", edgespacing=0.1, pspacing=0.2, verbose=False)
```
See the doc-strings and the notebook for more details. It is possible to define your own generation function and use it instead of the ones provided by `nbody`. It is also possible to collide particles on an individual basis
```
p = nbody.Particle(pos=(0, 0), vel=(1, 0),  mass=1, r=1)
t = nbody.Particle(pos=(3, 0), vel=(-1, 0), mass=1, r=1)

dt = p.dt2Hit(t)
p.move(dt)
p.collide2particle(t)
```
The simulation basically runs in the same way except additionally it keeps track of which collision should happend first by using a priority queue. See the notebook for more details on how exactly this is done.

The file `thermo_sim_examples.tar.bz2` should contain all the examples required to succesfully execute the notebook once untared. Functionality to read the snaphsots or sets of snapshots by `snapshot_data` and `sim_data` functions. Otherwise the following should work as well.
```
import nbody 
import matplotlib.pyplot as plt

# create a test system and a test snapshot
s = nbody.TDMSystem(10, 1)
s.snapshot(0)

# read the snapshot data in
snapshot = nbody.snapshot_data("snapshot_0.0")

# use one of the plotting utilities provided by nbody
fig, ax = plt.subplots()
nbody.plot_positions(ax, snapshot["x"], snapshot["y"])
plt.show()
```
