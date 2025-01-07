# Detectors

import os
import numpy as np
from . import surface,utils
from .line import Point,Vector,NormalVector,Line,Ray

class Detector(Optics):

    # This is a nondescript detector that records where each ray hits
    # and the energy (and normal?)
    
    def __init__(self,position,normal,vertices):
        self.position = Point(position)
        self.normal = NormalVector(normal)
        # construct plane object
        self.plane = surface.Plane.fromnormalcenter(normal,position)
        self.vertices = vertices
        self.data = []

    def __call__(self,ray):
        """ Process a ray """
        pass

    def intersections(self,ray):
        """ Get the intersections of a ray with the detector """
        tpnt = self.plane.intersections(ray)
        # now make sure it's within the vertices
    
    def display(self,bins=1000):
        """ Display an image of the recorded rays """
        pass
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Create a grid of points
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        X, Y = np.meshgrid(x, y)
        a,b,c,d = self.data
        # Calculate the corresponding Z values for the plane
        Z = (-d - a * X - b * Y) / c
        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=alpha)
        #ax.plot(pos[:,0],pos[:,1],pos[:,2],color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax


class CCD(Optics):

    def __init__(self,position,normal,vertices,nx=2048,ny=2048,pixsize=15e-6):
        self.position = Point(position)
        self.normal = NormalVector(normal)
        # construct plane object
        self.plane = surface.Plane.fromnormalcenter(normal,position)
        self.vertices = vertices
        self.nx = nx
        self.ny = ny
        self.pixsize = pixsize

    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Create a grid of points
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        X, Y = np.meshgrid(x, y)
        a,b,c,d = self.data
        # Calculate the corresponding Z values for the plane
        Z = (-d - a * X - b * Y) / c
        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=alpha)
        #ax.plot(pos[:,0],pos[:,1],pos[:,2],color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
