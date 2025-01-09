# Set up an optical layout/system

import os
import numpy as np
from . import optics

EPSILON = 1e-10

class Layout(object):

    def __init__(self,elements=None):
        self.__elements = []
        self.elements = elements

    @property
    def elements(self):
        """ Return the elements """
        return self.__elements

    @elements.setter
    def elements(self,value):
        # if it's a list, then iteration of the list
        if isinstance(value,list)==False and isinstance(value,tuple)==False:
            value = [value]
        for i,elem in enumerate(value):
            # Check that this is a valid optical element
            #if elem.__class__ not in optics.valid_optics:
            #    raise ValueError('Must be a valid optical element')
            self.__elements.append(elem)

    def append(self,value):
        self.elements = value
            
    @property
    def nelements(self):
        return len(self.elements)

    def __len__(self):
        return self.nelements
    
    def __call__(self,rays,verbose=False):
        """ Process multiple light rays through the system """
        if isinstance(rays,list)==False and isinstance(rays,tuple)==False:
            rays = [rays]
        nrays = len(rays)
        # Loop over rays
        newrays = nrays*[None]
        for i in range(nrays):
            newrays[i] = self.processray(rays[i],verbose=verbose)
        if nrays==1:
            newrays = newrays[0]
        return newrays
        # do we even need to return the rays since they are changed in place
        
    def processray(self,ray,verbose=False):
        """ Process a single light ray through the system """
        # Loop since there can be multiple reflections/refractions
        #   stop when the ray does not intersect any elements anymore
        count = 0
        intersects = self.intersections(ray)
        while len(intersects)>0 and ray.state=='inflight':
            # Exclude the surface that we are currently on, distance=0
            if count>0:
                intersects = [idata for idata in intersects if np.abs(idata[2])>EPSILON]
            # Determine which element it will hit first
            if len(intersects)==1:
                bestind = 0
                nextindex = intersects[bestind][0]
                nextpnt = intersects[0][1]
            elif len(intersects)>1:
                # Find distances
                dists = [i[2] for i in intersects]
                dists = np.array(dists).squeeze()
                bestind = np.argmin(dists)
                nextpnt = intersects[bestind][1]
                indexes = [i[0] for i in intersects]
                nextindex = indexes[bestind]
                #import pdb; pdb.set_trace()
            else:
                break
            # Process the ray through the next optical element
            nextelem = self[nextindex]
            if verbose:
                print(count+1,nextindex,nextpnt)
            ray = nextelem(ray)
            # Find new set of intersections
            intersects = self.intersections(ray)
            count += 1
        return ray

    def intersections(self,ray):
        """ Get all of the intersections of the ray with all of the elements """
        if len(self)==0:
            return []
        points = []
        # Loop over the elements and find the intersections
        #  keep track of which element it came from
        for i,elem in enumerate(self):
            if ray is None:
                print('layout.intersections() ray is None')
                import pdb; pdb.set_trace()
            intpnt = elem.intersections(ray)
            if len(intpnt)>0:
                if isinstance(intpnt,list)==False and isinstance(intpnt,tuple)==False:
                    intpnt = [intpnt]
                # Loop over the intersection points, the each get their own entry
                for j in range(len(intpnt)):
                    pt = intpnt[j]
                    # Get distance to the intersection point
                    dist = ray.distance(pt)
                    points.append((i,pt,dist))
        return points
        
    def __getitem__(self,index):
        if isinstance(index,int)==False and isinstance(index,slice)==False:
            raise ValueError('index must be an integer or slice')
        if isinstance(index,int) and index>len(self)-1:
            raise IndexError('index '+str(index)+' is out of bounds for axis 0 with size '+str(len(self)))
        if isinstance(index,int):
            return self.elements[index]
        # slice the list and return

    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < len(self):
            self._count += 1            
            return self[self._count-1]
        else:
            raise StopIteration

    def __repr__(self):
        """ Print out the elements """
        s = 'Layout({:d} elements)\n'.format(self.nelements)
        for e in self:
            s += str(e)+'\n'
        return s
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot of all of the elements """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # elements loop
        extents = []
        for i,elem in enumerate(self):
            elem.plot(ax=ax,color=color,alpha=alpha)
            #extents.append(elem.extent)

        # xr = [np.min(coords[:,0]),np.max(coords[:,0])]
        # dx = np.maximum(np.ptp(xr)*0.1,1)
        # xr = [xr[0]-dx,xr[1]+dx]
        # yr = [np.min(coords[:,1]),np.max(coords[:,1])]
        # dy = np.maximum(np.ptp(yr)*0.1,1)
        # yr = [yr[0]-dy,yr[1]+dy]
        # zr = [np.min(coords[:,2]),np.max(coords[:,2])]
        # dz = np.maximum(np.ptp(zr)*0.1,1)
        # zr = [zr[0]-dz,zr[1]+dz]
        xr0,yr0,zr0 = ax.get_xlim(),ax.get_ylim(),ax.get_zlim()
        # fxr = [min(xr[0],xr0[0]),max(xr[1],xr0[1])]
        # fyr = [min(yr[0],yr0[0]),max(yr[1],yr0[1])]
        # fzr = [min(zr[0],zr0[0]),max(zr[1],zr0[1])]
        # ax.set_xlim(fxr)
        # ax.set_ylim(fyr)
        # ax.set_zlim(fzr)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax


class Enclosure(object):

    # opaque enclosure surrounding the whole optical system
    
    def __init__(self,vertices):
        self.vertices

    def __call__(self,ray):
        pass
        
    def intersections(self,ray):
        pass

    def plot(self):
        pass
