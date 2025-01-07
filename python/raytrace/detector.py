# Detectors

import os
import numpy as np
from . import utils,surface,solidobject,lightray
from .line import Point,Vector,NormalVector,Line,Ray

class Detector(object):

    # This is a nondescript "ideal" detector that records where each ray hits
    # and the energy (and normal?)

    def __init__(self,vertices,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)
        if normal is None:
            normal = [0.0,0.0,1.0]
        self.normal = NormalVector(normal)
        self.vertices = vertices
        # construct plane object
        self.rectangle = surface.Rectangle.fromvertices(vertices)
        self.__data = []
        
    @property
    def detections(self):
        return self.__data

    @detections.setter
    def detections(self,value):
        self.__data.append(value)

    @property
    def coords(self):
        """ Return the coordinates of the detections """
        if len(self)==0:
            return None
        else:
            coo = np.atleast_2d([d[2] for d in self.detections])
            if len(self)==1:
                coo = coo.squeeze()
            return coo
        
    def __len__(self):
        return len(self.detections)
        
    def __call__(self,rays):
        """ Process a ray """
        if utils.sisinstance(rays,'raytrace.lightray.LightRay'):
            rays = [rays]
        for i in range(len(rays)):
            ray = rays[i]
            # check if it intersects
            tpnt = self.intersections(ray)
            if len(tpnt)>0:
                pnt = Point(tpnt).toframe(self)
                # (ray, point of detection in 3D space, point of detection in detector frame)
                self.detections = (ray.copy(),tpnt,pnt.data[:2])
                # Change the status of the ray
                ray.ray = Ray(position=tpnt,normal=ray.normal)
                ray.state = 'detected'

    def __repr__(self):
        dd = (len(self),*self.position.data,*self.normal.data)
        s = 'Detector({:d} detections, o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
        return s
                
    def intersections(self,ray):
        """ Get the intersections of a ray with the detector """
        tpnt = self.rectangle.intersections(ray)
        return tpnt
    
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


class CCD(Detector):

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
