import os
import numpy as np
from . import surface

# class for light rays

cspeed = 2.99792458e8  # speed of light in m/s
hplanck = 6.626e-34    # planck's constant, in J*s

class Ray(object):

    # Needs to keep track of the past points and normals
    
    def __init__(self,wavelength,position,normal):
        # wavelength in meters
        self.__wavelength = wavelength
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)
        self.history = []

    @property
    def wavelength(self):
        return self.__wavelength

    @property
    def frequency(self):
        # c = wave*frequency
        # frequency in hertz
        return cspeed/self.wavelength

    @property
    def energy(self):
        # E = h*f
        return hplanck*self.frequency
    
