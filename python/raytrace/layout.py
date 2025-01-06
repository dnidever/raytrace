# Set up an optical layout/system

import os
import numpy as np
from . import optics

class Layout(object):

    def __init__(self,elements=None):
        self.__elements = []

    @property
    def elements(self):
        """ Return the elements """
        return self.__elements

    @elements.setter
    def elements(self,value):
        # if it's a list, then iteration of the list
        if isinstance(value,list)==False and isinstance(value,tuple)==False:
            value = [value]
        for i in range(len(value)):
            # Check that this is a valid optical element
            if value.__class__ not in optics.valid_optics:
                raise ValueError('Must be a valid optical element')
            self.__elements.append(value)

    @property
    def nelements(self):
        return len(self.elements)

    def __len__(self):
        return self.nelements
    
    def __class__(self,rays):
        """ Process multiple light rays through the system """
        if isinstance(rays,list)==False and isinstance(rays,tuple)==False:
            rays = [rays]
        nrays = len(rays)
        # Loop over rays
        newrays = nrays*[None]
        for i in range(nrays):
            newrays.append(self.processray(rays[i]))
        if nrays==1:
            newrays = newrays[0]
        return newrays
        # do we even need to return the rays since they are changed in place
        
    def processray(self,ray):
        """ Process a single light ray through the system """
        # Loop since there can be multiple reflections/refractions
        #   stop when the ray does not intersect any elements anymore
        intersects = self.intersections(ray)
        while len(intersects)>0:
            # Determine which element it will hit first
            # find distances
            dists = [i[2] for i in intersects]
            import pdb; pdb.set_trace()
            
            intersects = self.intersections(ray)
        
    def intersections(self,ray):
        """ Get all of the intersections of the ray with all of the elements """
        if len(self)==0:
            return []
        points = []
        # Loop over the elements and find the intersections
        #  keep track of which element it came from
        for i,elem in enumerate(self):
            intpnt = elem.intersections(ray)
            if len(intpnt)>0:
                # get distances to the points
                dists = [ray.distance(p) for p in intpnt]
                points.append((i,intpnt,dists))
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
    
