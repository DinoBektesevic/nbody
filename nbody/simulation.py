from os.path import join
import heapq

import numpy as np

from .wall import Wall
from .particle import Particle
from .event import Event

__all__ = ["random_positions", "ordered_positions", "random_velocities",
           "identical_velocities", "ekin_constrained_velocities", "TDMSystem"]

def random_positions(Nparticles, Pr, edgespacing=None, **kwargs):
    """
    Sets the intiial particle location to completely random
    non-overlapping positions.
    """
    edgespace = Pr if edgespacing is None else edgespacing
    scale   = 10000 if kwargs.get("scale") is None else kwargs.get("scale")
    verbose = False if kwargs.get("verbose") is None else kwargs.get("verbose")
    if verbose:
        updated = False
        print("    2.2) Finding non-overlapping particle positions.")


    genran = lambda : np.random.randint(edgespace*scale,
                                        (1-edgespace)*scale)/scale
    x = np.array([genran()])
    y = np.array([genran()])
    while len(x) < Nparticles:
        if verbose and updated:
            updated = False
            curprogress = len(x)/(Nparticles)*100
            if curprogress % 10 == 0:
                print("        {0}%".format(curprogress))

        tmpx = genran()
        tmpy = genran()

        dx = x-tmpx
        dy = y-tmpy
        d  = np.sqrt(dx*dx + dy*dy)

        dontoverlap = d > 2.0*Pr
        if all(dontoverlap):
            x = np.append(x, tmpx)
            y = np.append(y, tmpy)
            updated = True

    return x, y

def ordered_positions(Nparticles, Pr, edgespacing=0.1, pspacing=0.1, **kwargs):
    """
    Sets the intiial particle location in an incremental order from left to
    right, top to bottom.
    """
    verbose = False if kwargs.get("verbose") is None else kwargs.get("verbose")
    edgespace = edgespacing if kwargs.get("edgespace") is None else kwargs.get("edgespace")
    partspace = pspacing if kwargs.get("partspace") is None else kwargs.get("partspace")
    if verbose: print("    2.2) Finding non-overlapping particle "+
                           "positions.")

    # set the first position by hand as the spacing from the edge + radii of
    # the particle 
    x, y = [], []
    x.append(edgespace+Pr)
    y.append(edgespace+Pr)
    while len(x) < Nparticles:
        if verbose:
            curprogress = len(x)/(Nparticles)*100
            if curprogress % 10 == 0:
                print("        {0}%".format(curprogress))

        # new position in the same row is the previous x position offset by
        # 2R to a case when they're touching, then by particle spacing  and
        # then by number of particles already placed in the row
        newx = x[-1] + (2.0*Pr+partspace)
        newy = y[-1]

        # if they're touching the outer edge, move up a row and get the initial
        # particle position
        if (newx) >= 1.0-Pr:
            newx = edgespace+Pr
            newy = y[-1] + (2.0*Pr+partspace)

        x.append(newx)
        y.append(newy)
    return x, y

def random_velocities(Nparticles, scaling=10000., **kwargs):
    """
    Returns 2 arrays of random velocities vx and vy respectively in the range
    [-v, v].
    """
    scale = scaling if kwargs.get("scale") is None else kwargs.get("scale")
    v = 1.0 if kwargs.get("v") is None else kwargs.get("v")

    vx = v*np.random.randint(-scale, scale, size=(Nparticles, ))/scale
    vy = v*np.random.randint(-scale, scale, size=(Nparticles, ))/scale
    return vx, vy

def identical_velocities(Nparticles, **kwargs):
    """
    Returns 2 arrays of identical velocitie components vx, vy respectively.
    Magnitude of the velocity components is v.
    """
    v = 1.0 if kwargs.get("v") is None else kwargs.get("v")
    vx = v*np.ones((Nparticles, ))
    vy = v*np.ones((Nparticles,))
    return vx, vy

def ekin_constrained_velocities(Nparticles, Pm=1, Ekin=1, **kwargs):
    """
    Returns 2 arrays of velocities in the vx, vy respectively. The
    magnitudes of the velocity components are such that they satisfy
    (0.5vx+0.5)**2 = v**2.
    Directions of the velocities are random.
    """
    ekin = Ekin if kwargs.get("pkin") is None else kwargs.get("pkin")
    m = Pm if kwargs.get("pm") is None else kwargs.get("pm")
    v = np.sqrt(2.0*ekin/m)
    vmag = v/2.0
    vx = vmag*np.random.choice([1., -1.], size=Nparticles)
    vy = vmag*np.random.choice([1., -1.], size=Nparticles)
    return vx, vy



class TDMSystem:
    """
    Defines a TDM system with N_particles, particle masses and radii.
    TDMSystem can be simulated up to the total simulation time in an event driven
    manner. The TDM system is described by the positions and velocities of
    particles. A series of Events is initialized which represent future particle
    collisions, provided the microstate of the system remains unchanged.
    Events are stored in a heap. A heap is an array with an invariant
        a[k] <= a[2*k+1] and
        a[k] <= a[2*k+2].
    Sorting a heap preserves this invariant. Zeroth element of a heap is always
    the smallest element of the heap.

    Because two Events are compared on the basis of the time at which they occur
    the zeroth element of the heap will always be the earliest Event that will
    occur.

    System's microstate is then evolved to the time of the first Event
    displacing all particles by their respective current velocities. Event, i.e.
    collision, is executed and involved particle states are changed.
    New events are generated for involved Event objects, when applicable, and
    inserted into the heap. Old events, created prior executing the Event, that
    involve these two objects will now declare as not valid.
    Events that would occur past the total simulated time, or occur with a time
    step that is larger than heapt, are not inserted into the heap.
    Simulation is executed until there are no Events left to execute.

    init parameters
    ----------------
       N_particles: number of particles to simulate
       total_t:     time up to which the simulation will execute
       p_mass:      mass of particles, same for all particles
       p_r:         radii of particles, same for all particles

    attributes
    ----------------
       t:         current simulation time step
       totalt:    total time up to which simulation will execute
       eventq:    heap of all Events set to happen in the simulation
       particles: list of all Particles objects in the simulation
       walls:     list of all the Wall objects defining the simulation geometry

    methods
    ----------------
       updateEventHeap(particle): for a given particle generate all possible
                                  events that would occur if system state didn't
                                  change.
       simulate():                updates all particle states up to event time
                                  and executes event, repeats until total simu-
                                  lation time doesn't exceed totalt.

    """
    def __init__(self, N_particles, t_total,
                 p_mass=1, p_r=0.01, p_v=1.0, p_kin=1.0,
                 pos_type="random", pos_func=None,
                 vel_type="random", vel_func=None,
                 verbose=True, save_t=0.01, heap_t=None, **kwargs):
        """
        Return a TDM System with N particles with mass and radii.

        N_particles: number of particles to simulate
        p_mass:      particle mass
        p_r:         particle radius
        t_total:     simulation end time
        """
        self.nparticles = N_particles
        self.pm         = p_mass
        self.pr         = p_r
        self.pv         = p_v
        self.pkin       = p_kin

        self.verbose = verbose

        self.totalt  = t_total
        self.savet   = save_t
        self.heapt   = save_t if heap_t is None else heap_t
        self.postype = pos_type
        self.posfunc = pos_func
        self.veltype = vel_type
        self.velfunc = vel_func
        self.t       = 0
        self.eventq  = []

        self.__initGeometry()
        self.__initParticles(**kwargs)
        self.__initEventHeap()



    def __initGeometry(self):
        """
        Initialize systems geometry - a box.
        """
        if self.verbose: print("1) Initializing geometry.")
        self.walls = [Wall("horizontal"), Wall("horizontal"),
                      Wall("vertical"), Wall("vertical")]

    def __initParticles(self, **kwargs):
        """
        Initialize particles to random positions and velocities.
        """
        if self.verbose: print("2) Initializing particles.")

        oom = int(np.log10(self.nparticles))
        scale = oom*10
        if scale == 0: scale=1

        xlen = abs(1-0) #self.walls in a general case
        ylen = abs(1-0)
        totsurface = xlen*ylen
        if self.nparticles*(np.pi*self.pr**2) > totsurface:
            errstr = "Can not fit {0} particles with radii {1} "+\
                     "in a box of area {2}."
            raise ValueError(errstr.format(self.nparticles, self.pr, totsurface))



        ##############################################################
        #                       Velocities
        ##############################################################
        if self.verbose: print("    2.1) Initializing particle velocities.")
        if self.veltype == "identical":
            vx, vy = identical_velocities(self.nparticles, verbose=self.verbose,
                                          **kwargs)
        elif self.veltype == "ekin_constrained":
            vx, vy = ekin_constrained_velocities(self.nparticles, pkin=self.pkin,
                                                 pm=self.pm, verbose=self.verbose,
                                                 **kwargs)
        elif self.veltype == "user_defined":
            vx, vy = self.velfunc(self.nparticles, p_r=self.pr, p_v=self.pv,
                                  p_mass=self.pm, p_kin=self.pkin,
                                  verbose=self.verbose, **kwargs)
        else:
            vx, vy = random_velocities(self.nparticles, v=self.pv,
                                       verbose=self.verbose, **kwargs)
        if self.verbose: print("        Done.")

        ##############################################################
        #                       Positions
        ##############################################################
        if self.verbose: print("    2.1) Initializing particle positions.")
        if self.postype == "ordered":
            x, y = ordered_positions(self.nparticles, self.pr,
                                     verbose=self.verbose, **kwargs)
        elif self.postype == "user_defined":
            x, y = self.posfunc(self.nparticles, self.pr, p_v=self.pv,
                                p_mass=self.pm, p_kin=self.pkin,
                                verbose=self.verbose, **kwargs)
        else:
            x, y = random_positions(self.nparticles, self.pr,
                                    verbose=self.verbose, **kwargs)
        if self.verbose: print("        Done.")


        ##############################################################
        #                       Particles
        ##############################################################
        if self.verbose: print("    2.3) Instatiating Particles.")
        self.particles = list(map(Particle, zip(x, y), zip(vx, vy),
                                  [self.pr]*self.nparticles,
                                  [self.pm]*self.nparticles))
        if self.verbose: print("        Done.")



    def __initEventHeap(self):
        """
        For consistency, creates an initial Event heap.
        """
        if self.verbose: print("3) Initializing Event Heap.")
        i = 0

        for p1 in self.particles:
            self.updateEventHeap(p1)

            if self.verbose:
                curprogress = i/(self.nparticles)*100
                if curprogress % 10 == 0:
                    print("    {0}%".format(curprogress))
                i+=1
        if self.verbose: print("    Done.")

    def __repr__(self):
        m = self.__class__.__module__
        n = self.__class__.__name__
        return "<{0}.{1}(Nparticles={2}, Nevents={3})>".format(m, n,
                                                               self.nparticles,
                                                               len(self.eventq))

    def updateEventHeap(self, p1):
        """
        Update the event heap with all possible events generated for particle.

        p1: particle for which all future events need to be created.
        """
        for p2 in self.particles:
            dt = p1.dt2Hit(p2)
            if (self.t+dt <= self.totalt):
                e = Event(p1, p2, self.t+dt)
                heapq.heappush(self.eventq, e)

        #particle wall collisions
        dthor = p1.dt2HitWall(self.walls[0])
        dtver = p1.dt2HitWall(self.walls[2])
        if (self.t+dthor <= self.totalt):
            heapq.heappush(self.eventq, Event(p1, self.walls[0], self.t+dthor))
        if (self.t+dtver <= self.totalt):
            heapq.heappush(self.eventq, Event(p1, self.walls[2], self.t+dtver))


    def simulate(self, snapshot=True):
        """
        As long as the event heap is not empty, evolves the particle states to
        the first event in the heap, executes the event/collision, removes the
        event from the heap, updates the heap with new events, unvalidates old
        events for particles involved in the collision and repeats.
        Handles data output as well.

        snapshot_dt: minimal time span between two snapshots.
        """
        snap_t = 0
        while len(self.eventq) > 0:
            e = heapq.heappop(self.eventq)

            if e.isValid():
                #evolve system to the time of collission
                for p in self.particles:
                    p.move(e.t-self.t)
                self.t = e.t
                e.execute()

                if e.obj1type == "Particle":
                    self.updateEventHeap(e.obj1)
                if e.obj2type == "Particle":
                    self.updateEventHeap(e.obj2)

                if snapshot and (self.t-snap_t)>self.savet:
                    self.snapshot(self.t)
                    snap_t = self.t

    def snapshot(self, t):
        """
        Create a snapshot of current microstate.
        """
        with open("snapshot_{0:.10}".format(float(t)), "w") as out:
            out.write("#t x y vx vy m r ncollisions\n")
            for p in self.particles:
                outstr = "{0} {1} {2} {3} {4} {5} {6} {7}\n"
                outstr = outstr.format(self.t, p.x, p.y, p.vx, 
                                       p.vy, p.m, p.r, p.ncollisions)
                out.write(outstr)
