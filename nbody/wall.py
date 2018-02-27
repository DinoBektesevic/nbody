class Wall:
    """
    Describes a infinitely long wall with an horizontal or vertical orientation.
    The wall currently actually represents 2 walls such that 2 wall instances
    are sufficient to describe a box with boundaries [0, 1]. Usage:
        w = Wall("horizontal")

    init parameters
    ----------------
    orientation: string, either "horizontal" or "vertical".
    """
    def __init__(self, orientation):
        """
        Infinitely long wall with an horizontal or vertical orientation.

        orientation: string, "horizontal" or "vertical".
        """
        self.orientation = orientation

    def __repr__(self):
        m = self.__class__.__module__
        n = self.__class__.__name__
        return "<{0}.{1}(orientation={2})>".format(m, n, self.orientation)
