# Optical surface

import os
import numpy as np
import copy
from . import utils

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

    @property
    def rotation(self):
        """ Return the rotation matrix."""

        # Example (using ZYX convention):
        # Given a normal vector: n = [nx, ny, nz]
        # Calculate the rotation matrix:
        # R = [ [cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll),
        #        cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)],
        #       [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll),
        #        sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll)],
        #       [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)] ]
        # Extract Euler angles:
        # pitch = arcsin(-R[2, 0])
        # yaw = atan2(R[1, 0], R[0, 0])
        # roll = atan2(R[2, 1], R[2, 2])
        r = self.r
        data = self.data/self.r
        rot = np.zeros((3,3),float)

        #rot[0,:] = [data[0]*data[2], data[1]*data[2], -1]
        #rot[1,:] = [-data[1], data[0], 0.0]
        #rot[2,:] = data

        #rot[0,:] = [-data[1], data[0], 0.0]
        #rot[1,:] = [data[0]*data[2], data[1]*data[2], -1]
        #rot[2,:] = data
        
        rot[0,:] = [self.x*self.z/r, self.y*self.z/r, -r]
        rot[1,:] = [-self.y/r, self.x/r, 0.0]
        rot[2,:] = [self.x, self.y, self.z]

        #rot[:,0] = [self.x*self.z/r, self.y*self.z/r, -r]
        #rot[:,1] = [-self.y/r, self.x/r, 0.0]
        #rot[:,2] = [self.x, self.y, self.z]
        
        #rot[0,:] = [self.y/r, -self.x/r, 0]
        #rot[1,:] = [self.x*self.z/r, self.y*self.z/r, -r]
        #rot[2,:] = [self.x, self.y, self.z]

        #alpha = np.arccos(-Z2/np.sqrt(1-Z3**2))
        alpha = np.arctan2(Z1,-Z2)
        eta = np.arccos(Z3)
        #gamma = np.arccos(Y3/np.sqrt(1-Z3**2))
        gamma = np.arctan2(X3,Y3)
        
        return rot
        
    
    def copy(self):
        return Vector(self.data.copy())

    
class NormalVector(Vector):
    """ Normal vector."""

    def __init__(self,data):
        if len(data)!=3:
            raise ValueError('data needs three elements')
        self.__data = np.array(data).astype(float)
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
        self.point = np.array(point).astype(float)
        self.slopes = np.array(slopes).astype(float)

    @property
    def point(self):
        return self.__point

    @point.setter
    def point(self,value):
        if len(value)!=3:
            raise ValueError('value needs three elements')
        pnt = np.array(value).astype(float)
        self.__point = pnt

    @property
    def slopes(self):
        return self.__slopes

    @slopes.setter
    def slopes(self,value):
        if len(value)!=3:
            raise ValueError('value needs three elements')
        slp = np.array(value).astype(float)
        self.__slopes = slp
    
    def __call__(self,t):
        """ Return the position at parametric value t """
        pos = np.zeros(3,float)
        pos[0] = self.point[0] + t*self.slopes[0]
        pos[1] = self.point[1] + t*self.slopes[1]
        pos[2] = self.point[2] + t*self.slopes[2]
        return Point(pos)

    @classmethod
    def frompoints(cls,p1,p2):
        """ Use two points to define the line. """
        pnt1 = np.atleast_1d(p1).astype(float)
        pnt2 = np.atleast_1d(p2).astype(float)
        point = pnt1
        slopes = (pnt2-pnt1)
        return Line(point,slopes)
    
    @property
    def data(self):
        return self.point,self.slopes

    def __array__(self):
        return self.data

    def __repr__(self):
        s = 'Line([(x,y,z)=({:.3f},{:.3f},{:.3f})+t({:.3f},{:.3f},{:.3f}))'.format(*self.point,*self.slopes)
        return s
    
    def copy(self):
        return Line(self.data.copy())


# circle
# ellipse
# cylinder
    
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
        """ Does the line intersect the surface """
        intpts = self.intersections(line)
        if len(intpts)==0:
            return False
        else:
            return True

    def intersection(self,ray):
        """ Return the first intersection point """
        pass

    def copy(self):
        return copy.deepcopy(self)
    

class Plane(Surface):

    def __init__(self,normal,d):
        self.normal = NormalVector(normal)
        self.d = d
        self.position = None

    @property
    def data(self):
        dt = np.zeros(4,float)
        dt[:3] = self.normal.data
        dt[3] = self.d
        return d
        
    @property
    def equation(self):
        s = '{:.3f}*x+{:.3f}*y+{:.3f}*z + {:.3f} = 0'.format(*self.normal,self.d)
        return s
    
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

    def intersection(self,ray):
        """ Return the first intersection point """
        pass
    
    
class Parabola(Surface):

    def __init__(self,a,position,normal):
        #super().__init__(**kw)
        # a is the leading coefficient of the parabola and determines the shape
        # negative a is convex
        # positive a is concave
        self.__a = float(a)
        if self.__a == 0:
            raise ValueError('a must be nonzero')
        self.position = Point(position)
        self.normal = NormalVector(normal)

        # parabola equation in 3D
        # z = a*(x**2 + y**2)

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self,value):
        self.__a = a

    @property
    def center(self):
        return self.position.data

    @property
    def vertex(self):
        return self.position.data
    
    @property
    def convex(self):
        # a is the leading coefficient of the parabola and determines the shape
        # negative a : convex
        # positive a : concave
        return self.a<0
        
    def distance(self,obj):
        if hasattr(obj,center):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        return np.linalg.norm(self.center-pnt)
        
    def intersections(self,line):
        """ Return the intersection points """
        # rotate the line into the Parabola reference frame
        # line in parametric form
        # substitute x/y/z in parabola equation for the line parametric equations
        # solve quadratic equation in t
        rline = line.rotate(self.normal)
        intpts = utils.intersect_line_parabola(line,self)
        return intpts

    @property
    def focus(self):
        """ Position of the focus."""
        pass

    @property
    def focal_length(self):
        """ Focal length """
        return 1/(4*np.abs(self.__a))

    @property
    def axis_of_symmetry(self):
        """ Return the axis of symmetry """
        return Line(self.center,self.normal.data)
    
    @property
    def directrix(self):
        """ Return the directrix. """
        pass
