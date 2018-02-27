from .particle import Particle

class Event:
    """
    Describes a TDM system event between two objects: particle-particle or
    particle-wall.
    Event is capable of verifying its own validity by tracking
    if any of the objects involved in the event were previously involved in an
    event that would change their state. This is why incrementing
    Particle.ncollisions is very important.
    Event can execute by calling the methods of involved objects in
    correct order.
    Events are compared on the basis of the time at which they occur.
    Usage:
        e = Event(Particle, Particle)
        e = Event(Particle, Wall)
        e.isValid()
        e.execute()

    init parameters
    ----------------
       obj1: Particle or Wall object involved in the Event
       obj2: Particle or Wall object involved in the Event
       t:    time from the start of simulation at which event happens.

    attributes
    ----------------
       obj1type:     type of object involved in the event. Used to determine
                     correct collision method order when event is executed.
       obj2type:     type of object involved in the event. Used to determine
                     correct collision method order when event is executed.
       initncolobj1: number of collisions of object 1 at the time of creation
                     of the event. Set to None when not applicable. Used to
                     determine the validity of the event.
       initncolobj2: number of collisions of object 1 at the time of creation
                     of the event. Set to None when not applicable. Used to
                     determine the validity of the event.
       t:            time since the start of the simulation at which this event
                     occurs.

    methods
    ----------------
    isValid(): returns True if the state of the objects remained unchanged since
               the time of creation and False otherwise.
    execute(): calls the appropriate collision methods based on the types of
               object 1 and 2.
    """
    def __init__(self, obj1, obj2, t):
        """
        Returns an Event containing the objects that participate in the event,
        their types, their current collision states and time of event measured
        from simulation start.

        obj1: one of the object participating in the event
        obj2: other object participating in the event
        t:    time of the event, measured since the start of simulation.
        """
        self.obj1type = obj1.__class__.__name__
        self.obj2type = obj2.__class__.__name__

        self.initncolobj1 = None
        self.initncolobj2 = None
        if self.obj1type == "Particle":
            self.initncolobj1 = obj1.ncollisions
        if self.obj2type == "Particle":
            self.initncolobj2 = obj2.ncollisions

        self.obj1 = obj1
        self.obj2 = obj2
        self.t = t

    def __eq__(self, that):
        if self.t == that.t:
            return True
        return False

    def __gt__(self, that):
        if self.t > that.t:
            return True
        return False

    def __lt__(self, that):
        return not self > that

    def __repr__(self):
        m = self.__class__.__module__
        n = self.__class__.__name__
        return """<{0}.{1}(obj1={2},
                    obj2={3},
                    t={4:.4f})>""".format(m, n, self.obj1, self.obj2, self.t)

    def isValid(self):
        """
        Return True if none of the object participated in a collision that would
        change their state since the creation of Event.
        """
        avalid, bvalid = False, False

        if (self.obj1type == "Particle" and
            self.obj1.ncollisions == self.initncolobj1):
            avalid = True
        elif self.obj1type == "Wall":
            avalid = True

        if (self.obj2type == "Particle" and 
            self.obj2.ncollisions == self.initncolobj2):
            bvalid = True
        elif self.obj2type == "Wall":
            bvalid = True

        
        return avalid and bvalid

    def execute(self):
        """
        Determines the appropriate collision method, based on object types,
        and executes it.
        """
        if self.obj1type == "Particle" and self.obj2type == "Particle":
            self.obj1.collide2particle(self.obj2)
        if self.obj1type == "Particle" and self.obj2type == "Wall":
            self.obj1.collide2wall(self.obj2)
        if self.obj2type == "Particle" and self.obj1type == "Wall":
            self.obj2.collide2wall(self.obj1)




