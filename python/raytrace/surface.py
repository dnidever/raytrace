# Optical surface

import os
import numpy as np
import copy

class Point(object):
    """ Class for a point."""

    def __init__(self,position):
        if len(position)!=3:
            raise ValueError('Point needs three elements')
        if isinstance(position,Point):
            pos = position.data.copy()
        else:
            pos = np.array(position).astype(float)
        self.data = pos

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self,value):
        if isinstance(value,Point):
            pos = value.data.copy()
        else:
            if len(value)!=3:
                raise ValueError('value needs three elements')
            pos = np.array(value).astype(float)
        self.__data = pos
    
    def __array__(self):
        return self.data

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    @property
    def r(self):
        return np.linalg.norm(self.data)
    
    def __repr__(self):
        s = 'Point(x={:.3f},y={:.3f},z={:.3f})'.format(*self.data)
        return s

    def copy(self):
        return Point(self.data.copy())
    
    # arithmetic operations
    def _check_value(self,value):
        """ Check that the input value is okay """
        if len(np.atleast_1d(value)) != 1 and len(np.atleast_1d(value)) != 3:
            raise ValueError('Value must have 1 or 3 elements')
            
    def __add__(self, value):
        if isinstance(value,Point):
            data = self.data + value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            data = self.data + value
        return Point(data)
        
    def __iadd__(self, value):
        if isinstance(value,Point):
            data = value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            data = value
        self.data += data
        return self
        
    def __radd__(self, value):
        return self + value
        
    def __sub__(self, value):
        newpoint = self.copy()
        if isinstance(value,Point):
            data = self.data
            newpoint.data -= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newpoint.data -= value
        return newpoint
              
    def __isub__(self, value):
        if isinstance(value,Point):
            self.data -= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data -= value
        return self

    def __rsub__(self, value):
        return self - value        
    
    def __mul__(self, value):
        newpoint = self.copy()
        if isinstance(value,Point):
            newpoint.data *= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newpoint.data *= value
        return newpoint
               
    def __imul__(self, value):
        if isinstance(value,Point):
            self.data *= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data *= value
        return self
    
    def __rmul__(self, value):
        return self * value
               
    def __truediv__(self, value):
        newpoint = self.copy()
        if isinstance(value,Point):
            newpoint.data /= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newpoint.data /= value
        return newpoint
      
    def __itruediv__(self, value):
        if isinstance(value,Point):
            self.data /= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data /= value
        return self

    def __rtruediv__(self, value):
        data = self.data.copy()
        if isinstance(value,Point):
            newdata = value.data / data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newdata = value / data
        return Point(newdata)

    def __lt__(self, b):
        return self.r < b.r
    
    def __le__(self, b):
        return self.r <= b.r
    
    def __eq__(self, b):
        return np.all(self.data == b.data)
    
    def __ne__(self, b):
        return np.any(self.data != b.data)
    
    def __ge__(self, b):
        return self.r >= b.r
    
    def __gt__(self, b):
        return self.r > b.r
    

class Vector(object):
    """ Vector """

    def __init__(self,data):
        self.data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self,value):
        if len(value)!=3:
            raise ValueError('value needs three elements')
        pos = np.array(value).astype(float)
        self.__data = pos
    
    def __array__(self):
        return self.data

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]
    
    def __repr__(self):
        s = 'Vector([{:.3f},{:.3f},{:.3f})'.format(*self.data)
        return s
    
    @property
    def r(self):
        return np.linalg.norm(self.data)

    @property
    def rho(self):
        """ length in x-y plane """
        return np.sqrt(self.data[0]**2+self.data[1]**2)

    @property
    def theta(self):
        """ Return the polar angle (angle from x-y plane) in degrees. """
        return np.rad2deg(np.arctan2(self.data[2],self.rho))

    @property
    def phi(self):
        """ Return the azimuthal angle (angle from positive x-axis) in degrees."""
        return np.rad2deg(np.arctan2(self.data[1],self.data[0]))
    
    def copy(self):
        return Vector(self.data.copy())

    
class NormalVector(Vector):
    """ Normal vector."""

    def __init__(self,data):
        if len(data)!=3:
            raise ValueError('data needs three elements')
        self.__data = np.array(data).astype(float)
        #super(NormalVector,self).__init__(data)
        self.__data /= np.linalg.norm(self.__data)   # normalize

    @property
    def data(self):
        return self.__data
        
    def __repr__(self):
        s = 'NormalVector([{:.3f},{:.3f},{:.3f})'.format(*self.data)
        return s
        
    def copy(self):
        return NormalVector(self.data.copy())

    
class Line(object):
    """ Line."""

    def __init__(self,point,slopes):
        if len(point)!=3:
            raise ValueError('point needs three elements')
        if len(slopes)!=3:
            raise ValueError('slopes needs three elements')
        # Parametric form for 3-D line
        # (x,y,z) = (x0,y0,z0)+t(a,b,c)
        self.__point = Point(point)
        self.__slopes = np.array(slopes).astype(float)

    def __call__(self,t):
        """ Return the position at parametric value t """
        pos = np.zeros(3,float)
        pos[0] = self.__point.x + t*self.__slopes[0]
        pos[1] = self.__point.y + t*self.__slopes[1]
        pos[2] = self.__point.z + t*self.__slopes[2]
        return Point(pos)
        
    @property
    def data(self):
        return self.__normal

    def __array__(self):
        return self.data
    
    def copy(self):
        return Line(self.data.copy())

    
class Surface(object):
    """ Main surface base class. """

    def __init__(self,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)
        if normal is None:
            normal = [0.0,0.0,1.0]
        self.normal = NormalVector(normal)

    @property
    def center(self):
        return self.position
    
    def distance(self,obj):
        """ Return distance of point/object to the center of the surface."""
        pass

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    def copy(self):
        return copy.deepcopy(self)
    

class Plane(Surface):

    def __init__(self,**kw):
        super().__init__(**kw)
        self.position = None

    def distance(self,obj):
        """ Distance from a point and the normal of the plane. """
        if hasattr(obj,'center'):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        # 
        return dist

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    
class Sphere(Surface):

    def __init__(self,radius,**kw):
        super().__init__(**kw)
        self.radius = radius
        self.convex = convex

    def distance(self,obj):
        if hasattr(obj,center):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        return np.linalg.norm(self.center-pnt)
    
    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

class HalfSphere(Surface):

    """ Half Sphere, flat on other (bottom) side """
    
    def __init__(self,radius,**kw):
        super().__init__(**kw)
        # negative radius is convex
        # positive radius is concave
        self.radius = radius

    @property
    def convex(self):
        return self.radius
        
    def distance(self,obj):
        if hasattr(obj,center):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        return np.linalg.norm(self.center-pnt)
    
    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass
    
    
class Parabola(Surface):

    def __init__(self,a,**kw):
        super().__init__(**kw)
        # a is the leading coefficient of the parabola and determines the shape
        # negative a is convex
        # positive a is concave
        self.__a = float(a)
        if self.__a == 0:
            raise ValueError('a must be nonzero')
        self.position = Point(position)
        self.normal = NormalVector(normal)

    @property
    def convex(self):
        return self.a<0
        
    def distance(self,obj):
        if hasattr(obj,center):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        return np.linalg.norm(self.center-pnt)

    def dointersect(self,ray):
        """ Does the ray intersect the the surface """
        pass

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    @property
    def focus(self):
        """ Position of the focus."""
        pass

    @property
    def focal_length(self):
        """ Focal length """
        return 1/(4*np.abs(self.__a))

    @property
    def directrix(self):
        """ Return the directrix. """
        pass
