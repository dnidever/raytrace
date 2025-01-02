# optical elements

import os
import numpy as np
from .surface import Point,NormalVector,Plane

class Optics(object):
    """ Main class for optical elements """

    def __init__(self,index_of_refraction=None,position=None,orientation=None):
        self.index_of_refraction = index_of_refraction
        self.position = position
        self.orientation = orientation

    def __call__(self,ray):
        """ process the light ray through the optical element."""
        pass
        
    def intersect(self,ray):
        """ does a ray intersect with us."""
        pass

    def intersectpoint(self,ray):
        """ find first point of intersection."""
        pass


class Lens(Optics):

    def __init__(self,topsurface,bottomsurface,radius,**kw):
        super().__init__(**kw)
        self.topsurface = topsurface
        self.bottomsurface = bottomsurface
        self.radius = radius


class Plane(Optics):

    def __init__(self,normal):
        super().__init__(**kw)
        self.normal = NormalVecetor(normal)


class Grating(Optics):

    def __init__(self,normal):
        super().__init__(**kw)
        self.normal = NormalVector(normal)


class Detector(Optics):

    def __init__(self,normal):
        super().__init__(**kw)
        self.normal = NormalVector(normal)


        
