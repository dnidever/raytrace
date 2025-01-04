# optical elements

import os
import numpy as np
from . import surface
from . import ray

class Optics(object):
    """ Main class for optical elements """

    def __init__(self,index_of_refraction=None,position=None,normal=None):
        self.index_of_refraction = index_of_refraction
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)

    def __call__(self,ray):
        """ Process the light ray through the optical element."""
        pass
        
    def intersect(self,ray):
        """ Does a ray intersect with us."""
        pass

    def intersectpoint(self,ray):
        """ Find first point of intersection."""
        pass

class FlatMirror(Optics):

    def __init__(self,index_of_refraction=None,position=None,normal=None):
        # Need plane and vertices
        if position is None:
            position = [0,0,0]
        if normal is None:
            normal = [0,0,1]
        self.index_of_refraction = index_of_refraction
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)
        # Construct a plane object
        self.plane = surface.Plane.fromnormalcenter(normal,position)

    def __call__(self,ray):
        """ Process a ray """
        # Get intersections
        tpnt = self.intersections(ray)
        # Find the angle relative to the surface+normal
        #  to get the reflection
        # Update the ray's path
        pass

    def reflection(self,ray,point):
        """ Figure out the reflection for a ray at a specific reflection point """
        # get the angle the ray makes with the surface
        # then reflect it
        # return the line of the reflected ray
        
        reflected_line = Line(point,norm_reflected)
        return reflected_line
    
    def dointersect(self,ray):
        """ Does a ray intersect with us."""
        pass
    
    def intersections(self,ray):
        """ Get the intersections of a ray with the detector """
        tpnt = self.plane.intersections(ray)
        # now make sure it's within the vertices
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot  """
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
        

class ConcaveMirror(Optics):

    def __init__(self,index_of_refraction=None,position=None,normal=None):
        self.index_of_refraction = index_of_refraction
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)

        # Need sphere or parabola and vertices
        
    def __call__(self,ray):
        """ Process the light ray through the optical element."""
        pass
        
    def intersect(self,ray):
        """ Does a ray intersect with us."""
        pass

    def intersectpoint(self,ray):
        """ Find first point of intersection."""
        pass

class ConvexMirror(Optics):

    def __init__(self,index_of_refraction=None,position=None,normal=None):
        self.index_of_refraction = index_of_refraction
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)

        # Need sphere or parabola and vertices
        
    def __call__(self,ray):
        """ Process the light ray through the optical element."""
        pass
        
    def intersect(self,ray):
        """ Does a ray intersect with us."""
        pass

    def intersectpoint(self,ray):
        """ Find first point of intersection."""
        pass
    
    
class Lens(Optics):

    def __init__(self,topsurface,bottomsurface,radius,**kw):
        super().__init__(**kw)
        self.topsurface = topsurface
        self.bottomsurface = bottomsurface
        # this radius is the extent of the lens
        self.radius = radius

    def __call__(self,ray):
        """ Process the light ray through the optical element."""
        pass
    
    def topintersections(self,ray):
        """ Return intersections of the top """
        tpnt = self.topsurface.intersections(ray)
        # impose radius
        return tpnt
        
    def bottomintersections(self,ray):
        """ Return intersections of the bottom """
        bpnt = self.bottomsurface.intersections(ray)
        # impose radius
        return bpnt
        
    def dointersect(self,ray):
        """ Does the ray intersect the surface """
        tpnt = self.topintersections(ray)
        bpnt = self.bottomintersections(ray)
        if len(tpnt)>0 or len(bpnt)>0:
            return False
        else:
            return True


class Grating(Optics):

    def __init__(self,normal):
        super().__init__(**kw)
        self.normal = surface.NormalVector(normal)

    def __call__(self,ray):
        """ Process a ray """
        pass

    def intersections(self,ray):
        """ Get the intersections of a ray with the grating """
        tpnt = self.plane.intersections(ray)
        # now make sure it's within the vertices
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot of grating """
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
        # Include the lines that show the grating lines
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
        

class Detector(Optics):

    # This is a nondescript detector that records where each ray hits
    # and the energy (and normal?)
    
    def __init__(self,position,normal,vertices):
        self.position = surface.Point(position)
        self.normal = surface.NormalVector(normal)
        # construct plane object
        self.plane = surface.Plane(normal,d)
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

    def __init__(self,normal,vertices,nx=2048,ny=2048,pixsize=15e-6):
        super().__init__(**kw)
        self.normal = surface.NormalVector(normal)
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
