# optical elements

import os
import numpy as np
from . import surface
from . import ray

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
        self.normal = surface.NormalVecetor(normal)


class Grating(Optics):

    def __init__(self,normal):
        super().__init__(**kw)
        self.normal = surface.NormalVector(normal)


class Detector(Optics):

    def __init__(self,normal,vertices):
        super().__init__(**kw)
        self.normal = surface.NormalVector(normal)
        self.vertices = vertices

    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot at input points t."""
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

    def __init__(self,normal,vertices,nx=2048,ny=2048,pixsize=15e-6):
        super().__init__(**kw)
        self.normal = surface.NormalVector(normal)
        self.vertices = vertices
        self.nx = nx
        self.ny = ny
        self.pixsize = pixsize

    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot at input points t."""
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
    
class Source(Optics):

    # Light source
    
    def __init__(self,position=None,kind='isotropic'):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = surface.Point(position)
        self.kind = kind

    @property
    def center(self):
        return self.position.data

    def __repr__(self):
        s = 'Source({:},o=[{:.3f},{:.3f},{:.3f}])'.format(self.kind,*self.center)
        return s
    
    def rays(self,n=1,wave=5000e-10):
        """ Make a number of random rays with input wavelengths """
        nwave = np.atleast_1d(wave).size
        if nwave==1:
            wave = np.zeros(n,float)+np.atleast_1d(wave)[0]
        rr = []
        if self.kind=='isotropic':
            u = np.random.rand(n)
            v = np.random.rand(n)
            theta = 2*np.pi*u
            phi = np.arccos(2*v-1)
            for i in range(n):
                normal = surface.NormalVector.fromangles(phi[i],theta[i])
                r = ray.Ray(wave[i],self.center,normal.data)
                rr.append(r)
        if len(rr)==1:
            rr = rr[0]
        return rr
