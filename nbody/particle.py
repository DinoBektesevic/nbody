import numpy as np

class Particle:
    """
    Describes a hard-sphere-like particle with position, velocity, mass and
    radius in 2D space.
    Usage:
        a = Particle(position, velocity, radius, mass)
        b = Particle((x, y), (vx, vy), r, m)
        dt = a.dt2Hit(b)
        a.move(dt)
        b.move(dt)
        a.collide2particle(b)

    init parameters
    ----------------
       pos: tuple of (x, y) coordinates that define the current position of the
             particle
       vel: tuple of (vx, vy) values that define the current particle velocities
            in x and y direction respectively
       r:   radius
       m:   mass

    attributes
    ----------------
       ncollisions: number of collisions particle has experienced so far,
                    **MUST** increment after every interaction

    methods
    ----------------
    dt2Hit(particle):           returns time required until collision with
                                particle occurs
    dt2HitWall(wall):           returns time required until collision with
                                wall occurs
    move(dt):                   propagates the current particle state
                                for time dt
    collide2particle(particle): collide two particles and change their state
    collide2wall(wall):         collide this particle to a wall and change its
                                state
    """
    def __init__(self, pos, vel, r, mass):
        """
        Returns a particle object with position, mass, radius and velocity in
        2D space.

        pos: tuple of (x, y) coordinates that define the current position of the
             particle
        vel: tuple of (vx, vy) values that define the current particle velocities
             in x and y direction respectively
        r:   radius
        m:   mass
        """
        self.x = pos[0]
        self.y = pos[1]

        self.vx = vel[0]
        self.vy = vel[1]

        self.r = float(r)
        self.m = float(mass)

        self.ncollisions = 0

    def __repr__(self):
        m = self.__class__.__module__
        n = self.__class__.__name__
        tmpstr = "<{0}.{1}(pos=({2:.2f}, {3:.2f}), vel=({4:.2f}, {5:.2f}))>"
        return tmpstr.format(m, n, self.x, self.y, self.vx, self.vy, self.ncollisions)

    def dt2Hit(self, that):
        """
        Returns time required to hit selected particle. If in current state the
        collision is not possible returned time is 'inf.'

        particle: a particle object for which time to collision
                  needs to be determined.
        """
        if(self == that):
            return float("inf")

        dx = that.x - self.x
        dy = that.y - self.y
        dr2 = dx*dx + dy*dy

        dvx = that.vx - self.vx
        dvy = that.vy - self.vy
        dv2 = dvx*dvx + dvy*dvy

        dvdr = dx*dvx + dy*dvy
        sigma = self.r + that.r

        d2 = (dvdr*dvdr) - dv2*(dr2-sigma*sigma)
        if(dvdr>=0) or (d2<0):
            return float('inf')

        return -(dvdr + np.sqrt(d2)) / dv2


    def dt2HitWall(self, wall):
        """
        Returns time required for this particle to hit the selected wall. If in
        current state the collision is not possible returned time is 'inf.'

        wall: a Wall object for which time to collision needs to be determined.
        """

        if wall.orientation == "vertical":
            if (self.vx>0):
                return ((1.0-self.x-self.r) / self.vx)
            elif (self.vx < 0):
                return ((self.r-self.x) / self.vx)

        elif wall.orientation == "horizontal":
            if (self.vy > 0):
                return ((1.0-self.y-self.r) / self.vy)
            elif (self.vy < 0):
                return ((self.r-self.y) / self.vy)

        return float('inf')

    def move(self, dt):
        """
        Evolve particle state for duration dt.

        dt: ammount of time for which current state will be propagated for.
        """
        self.x += self.vx*dt
        self.y += self.vy*dt


    def collide2particle(self, that):
        """
        Collide this particle with a particle. There are no checks performed to
        verify the collision is possible - responsibility to ensure collision
        conditions are valid lies on the user.

        that: particle with which current particle will collide.
        """
        self.ncollisions += 1
        that.ncollisions += 1

        dx  = that.x - self.x
        dy  = that.y - self.y
        dvx = that.vx - self.vx
        dvy = that.vy - self.vy

        dvdr = dx*dvx + dy*dvy
        #distance at collision is just the sum of radii
        dist = self.r + that.r

        magnitude = 2.0*self.m*that.m*dvdr / ((self.m+that.m)*dist)

        fx = magnitude*dx / float(dist)
        fy = magnitude*dy / float(dist)

        self.vx += fx / self.m
        self.vy += fy / self.m
        that.vx -= fx / that.m
        that.vy -= fy / that.m

    def collide2wall(self, wall):
        """
        Collide current particle with a wall. There are no checks performed to
        verify the collision is possible - responsibility to ensure collision
        conditions are valid lies on the user.

        wall: Wall instance with which current particle will collide.
        """
        self.ncollisions += 1
        if wall.orientation == "horizontal":
            self.vy = -self.vy
        elif wall.orientation == "vertical":
            self.vx = -self.vx
        else:
            raise ValueError("Expected wall orientation horizontal or vertical"+\
                             "got {0} instad.".format(wall.orientation))
