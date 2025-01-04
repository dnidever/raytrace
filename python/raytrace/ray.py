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
        self.wavelength = wavelength
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)
        self.path = []

    @property
    def wavelength(self):
        return self.__wavelength

    @wavelength.setter
    def wavelength(self,value):
        self.__wavelength = float(value)

    def __repr__(self):
        dd = (self.wavelength,*self.position.data,*self.normal.data)
        s = 'Ray(wave={:.3e},p=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
        return s
        
    @property
    def frequency(self):
        # c = wave*frequency
        # frequency in hertz
        return cspeed/self.wavelength

    @property
    def energy(self):
        # E = h*f
        return hplanck*self.frequency
    
    def plot(self,ax=None,color=None):
        """ Make a 3-D plot of the ray """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        x0,y0,z0 = self.position.data
        x = x0 + self.normal.data[0]*3
        y = y0 + self.normal.data[1]*3
        z = z0 + self.normal.data[2]*3
        ax.quiver(x0, y0, z0, x, y, z, arrow_length_ratio=0.1,color=color)
        ax.scatter(x0,y0,z0,color=color,s=20)
        ax.set_xlim(x0,x)
        ax.set_ylim(y0,y)
        ax.set_zlim(z0,z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
