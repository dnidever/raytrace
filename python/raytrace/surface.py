# Optical surface

import os
import numpy as np

class Point(object):
    """ Class for a point."""

    def __init__(self,position):
        if len(position)!=3:
            raise ValueError('Point needs three elements')
        self.__x = float(position[0])
        self.__y = float(position[1])
        self.__z = float(position[2])

    @property
    def data(self):
        return np.array([self.__x,self.__y,self.__z])

    
class NormalVector(object):
    """ Normal vector."""

    def __init__(self,normal):
        if len(position)!=3:
            raise ValueError('Normal needs three elements')
        self.__normal = np.array(normal).astype(float)
        self.__normal /= np.linalg.norm(self.__normal)   # normalize
        
    @property
    def data(self):
        return self.__normal

class Surface(object):
    """ Main surface class. """

    def __init__(self,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)
        if normal is None:
            normal = [0.0,0.0,1.0]
        self.normal = NormalVector(normal)

    def centerdistance(self,point):
        """ Return distance of point to the center of the surface."""
        pass

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass
    
class Plane(Surface):

    def __init__(self,**kw):
        super().__init(**kw)
        self.position = None

    def centerdistance(self,point):
        pass

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    
class Sphere(Surface):

    def __init__(self,radius,convex=True,**kw):
        super().__init(**kw)
        self.radius = radius
        self.convex = convex

    def centerdistance(self,point):
        pass

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    
class Parabola(Surface):

    def __init__(self,a,convex=True,**kw):
        super().__init(**kw)
        # a is the leading coefficient of the parabola and determines the shape
        self.__a = float(a)
        self.position = Point(position)
        self.normal = NormalVector(normal)
        self.convex = convex

    def centerdistance(self,point):
        pass

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass
