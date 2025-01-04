import os
import numpy as np
from . import surface

""" Light sources """

class Source(Optics):

    # Light source
    
    def __init__(self,position=None,kind=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = surface.Point(position)
        self.kind = kind

    @property
    def kind(self):
        return self.__kind

    @kind.setter
    def kind(self,value):
        self.__kind = str(value)
        
    @property
    def center(self):
        return self.position.data

    def __repr__(self):
        s = 'Source({:},o=[{:.3f},{:.3f},{:.3f}])'.format(self.kind,*self.center)
        return s
    
    def rays(self,n=1,wave=5000e-10):
        pass


class IsotropicSource(Optics):

    # Light source
    
    def __init__(self,position=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = surface.Point(position)
        self.kind = 'isotropic'

    @property
    def kind(self):
        return self.__kind

    @kind.setter
    def kind(self,value):
        self.__kind = str(value)
        
    @property
    def center(self):
        return self.position.data

    def __repr__(self):
        s = 'IsotropicSource({:},o=[{:.3f},{:.3f},{:.3f}])'.format(self.kind,*self.center)
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

    
class FiberSource(object):

    # fiber light source
    # has some f-ratio
    
    def __init__(self,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = surface.Point(position)
        self.kind = 'fiber'

    @property
    def kind(self):
        return self.__kind

    @kind.setter
    def kind(self,value):
        self.__kind = str(value)
        
    @property
    def center(self):
        return self.position.data

    def __repr__(self):
        s = 'FiberSource({:},o=[{:.3f},{:.3f},{:.3f}])'.format(self.kind,*self.center)
        return s
    
    def rays(self,n=1,wave=5000e-10):
        pass

    
class LaserSource(object):

    # laser light source
    # no f-ratio, emits rays with a single orientation
    
    def __init__(self,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = surface.Point(position)
        self.kind = 'laser'

    @property
    def kind(self):
        return self.__kind

    @kind.setter
    def kind(self,value):
        self.__kind = str(value)
        
    @property
    def center(self):
        return self.position.data

    def __repr__(self):
        s = 'LaserSource({:},o=[{:.3f},{:.3f},{:.3f}])'.format(self.kind,*self.center)
        return s
    
    def rays(self,n=1,wave=5000e-10):
        pass
