# optical elements

import os
import numpy as np
from . import surface
from .lightray import LightRay
from .line import Point,NormalVector,Line,Ray

class Optics(object):
    """ Main class for optical elements """

    def __init__(self,index_of_refraction=None,position=None,normal=None):
        self.index_of_refraction = index_of_refraction
        self.position = Point(position)
        self.normal = NormalVector(normal)

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

    def __init__(self,position=None,normal=None,vertices=None):
        # Need plane and vertices
        if position is None:
            position = [0,0,0]
        if normal is None:
            normal = [0,0,1]
        self.position = Point(position)
        self.normal = NormalVector(normal)
        self.vertices = vertices
        # Construct a plane object
        self.plane = surface.Plane.fromnormalcenter(normal,position)

    def __repr__(self):
        dd = (*self.position.data,*self.normal.data)
        s = 'FlatMirror(o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
        return s
        
    def __call__(self,ray):
        """ Process a ray """
        # Get intersections
        tpnt = self.intersections(ray)
        # No intersection, return original ray
        if len(tpnt)==0:
            return ray
        # Get the reflected ray
        reflected_ray = self.reflection(ray,tpnt)
        # Update the LightRay's ray
        #  and update it's path
        print(type(ray))
        #if isinstance(ray,LightRay):
        if hasattr(ray,'ray'):
            #print('this is a LightRay')
            ray.ray = reflected_ray
            # resetting "ray" also updates the history/path
            return ray
        # normal ray input, can't update anything
        else:
            #print('not a lightray')
            return reflected_ray
        
    def reflection(self,ray,point):
        """ Figure out the reflection for a ray at a specific point """
        # First we have to reverse the lightray's normal
        # and then reflect it about the mirror's normal
        mirror_normal = self.plane.normalatpoint(point)
        ray_normal = ray.normal.copy()
        # Flip the ray's normal, since it will be going "out"
        #   after it is reflected
        ray_flip_normal = -1*ray_normal
        # Then reflect it about the mirror normal
        ray_normal_reflected = ray_flip_normal.reflectabout(mirror_normal)
        reflected_ray = Ray(point,ray_normal_reflected)
        return reflected_ray

    def dointersect(self,ray):
        """ Does the ray intersect the surface """
        intpts = self.intersections(ray)
        if len(intpts)==0:
            return False
        else:
            return True

    def intersections(self,ray):
        """ Get the intersections of a ray with the detector """
        tpnt = self.plane.intersections(ray)
        # now make sure it's within the vertices
        #if self.vertices is not None:
        #import pdb; pdb.set_trace()
        return tpnt
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot  """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Create a grid of points
        a,b,c,d = self.plane.data
        if c != 0.0:
            x = np.linspace(-5, 5, 10)
            y = np.linspace(-5, 5, 10)
            X, Y = np.meshgrid(x, y)
            # Calculate the corresponding Z values for the plane
            Z = (-d - a * X - b * Y) / c
        elif b != 0.0:
            x = np.linspace(-5, 5, 10)
            z = np.linspace(-5, 5, 10)
            X, Z = np.meshgrid(x, z)
            # Calculate the corresponding Y values for the plane
            Y  = (-d - a * X - c * Z) / b
        elif a != 0.0:
            y = np.linspace(-5, 5, 10)
            z = np.linspace(-5, 5, 10)
            Y, Z = np.meshgrid(y, z)
            # Calculate the corresponding Y values for the plane
            X  = (-d - b * Y - c * Z) / a
        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=alpha)
        #ax.plot(pos[:,0],pos[:,1],pos[:,2],color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #import pdb; pdb.set_trace()
        return ax
        

class ConcaveMirror(Optics):

    def __init__(self,topsurface,position=None,normal=None):
        self.index_of_refraction = index_of_refraction
        self.position = Point(position)
        self.normal = NormalVector(normal)
        self.topsurface = topsurface
        
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

    def __init__(self,topsurface,position=None,normal=None):
        self.position = Point(position)
        self.normal = NormalVector(normal)
        self.topsurface = topsurface
        
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

    def __init__(self,position,normal):
        self.plane = surface.Plane.fromnormalcenter(normal,position)

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
