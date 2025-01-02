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

    
class Normal(object):
    """ Normal vector."""

    def __init__(self,normal):
        if len(position)!=3:
            raise ValueError('Normal needs three elements')
        self.__normal = np.array(normal).astype(float)

    @property
    def data(self):
        return self.__normal
    
class Plane(object):

    def __init__(self,normal):
        self.normal = Normal(normal)


class Sphere(object):

    def __init__(self,position,radius):
        self.position = Point(position)
        self.radius = radius
