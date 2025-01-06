# Optical surface

import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from . import utils,ray

EPSILON = 1e-10

def islinelike(obj):
    """ Check if an object is line-like """
    if isinstance(b,Line) or isinstance(b,ray.Ray) or issubclass(b.__class__,Vector):
        return True
    else:
        return False

def linelikenormal(obj):
    """ Return normal vector data of line-like object """
    if isinstance(b,Line):
        data = b.slopes.copy()
    elif isinstance(b,ray.Ray):
        data = ray.normal.data.copy()
    elif issubclass(b.__class__,Vector):
        data = b.data.copy()
    else:
        raise ValueError('Object is not line-like')
    return data

class Point(object):
    """ Class for a point."""

    def __init__(self,position):
        if isinstance(position,Point):
            position = position.data
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

    def distance(self,pnt):
        """ Return distance between this point and another point """
        if isinstance(pnt,Point):
            return np.linalg.norm(self.data-pnt.data)
        else:
            return np.linalg.norm(self.data-pnt)
    
    def __repr__(self):
        s = 'Point(x={:.3f},y={:.3f},z={:.3f})'.format(*self.data)
        return s

    def copy(self):
        return Point(self.data.copy())

    def toframe(self,obj):
        """ Return the point transformed to the frame of the input object."""
        if hasattr(obj,'normal')==False:
            raise ValueError('input object must have a normal')
        newdata = self.data.copy()
        if hasattr(obj,'center'):
            cen = obj.center
        elif hasattr(obj,'position'):
            cen = obj.position.data
        else:
            cen = np.zeros(3,float)
        newdata -= cen
        rot = obj.normal.rotation_matrix
        newdata = np.matmul(newdata,rot)
        newpnt = Point(newdata)
        return newpnt
    
    def plot(self,ax=None,color=None,size=50):
        """ Plot the point on an a plot. """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*self.data,color=color,s=size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
        
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
        if issubclass(data.__class__,Vector):
            data = data.data
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
    def theta(self,degrees=False):
        """ Return the polar angle (angle from positive z-axis) in degrees. """
        return np.rad2deg(np.arctan2(self.rho,self.data[2]))

    @property
    def phi(self,degrees=False):
        """ Return the azimuthal angle (angle from positive x-axis) in degrees."""
        return np.rad2deg(np.arctan2(self.data[1],self.data[0]))

    def dot(self,value):
        """ Return dot product of two vectors """
        if issubclass(value.__class__,self.__class__):
            data = value.data
        else:
            data = np.atleast_1d(value).astype(float)
            if data.size != 3:
                raise ValueError('must have three values')
        return np.dot(self.data,data)

    def cross(self,value):
        """ Return cross product of two vectors """
        if issubclass(value.__class__,self.__class__):
            data = value.data
        else:
            data = np.atleast_1d(value).astype(float)
            if data.size != 3:
                raise ValueError('must have three values')
        cp = np.cross(self.data,data)
        return self.__class__(cp)

    def anglebetween(self,value,degrees=False):
        """ Return the angle between two vectors """
        if issubclass(value.__class__,self.__class__):
            data = value.data
        else:
            data = np.atleast_1d(value).astype(float)
            if data.size != 3:
                raise ValueError('must have three values')
        dp = np.dot(self.data,data)
        r1 = np.linalg.norm(self.data)
        r2 = np.linalg.norm(data)
        if r1 < EPSILON or r2 < EPSILON:
            return 0.0
        angle = np.arccos(dp/(r1*r2))
        if degrees:
            angle = np.rad2deg(angle)
        return angle
    
    def isparallel(self,value):
        """ Check if a vector is parallel """
        if issubclass(value.__class__,self.__class__):
            data = value.data
        else:
            data = np.atleast_1d(value).astype(float)
            if data.size != 3:
                raise ValueError('must have three values')
        # if the cross-product of two vectors is zero, then they are parallel
        cp = np.cross(self.data,data)
        if np.linalg.norm(cp) < EPSILON:
            return True
        else:
            return False

    def isperpendicular(self,value):
        """ Check if a vector is perpendicular """
        if issubclass(value.__class__,self.__class__):
            data = value.data
        else:
            data = np.atleast_1d(value).astype(float)
            if data.size != 3:
                raise ValueError('must have three values')
        # calculate dot product, zero if they orthogonal
        dotproduct = np.dot(self.data,data)
        if np.abs(dotproduct) < EPSILON:
            return True
        else:
            return False

    def reflectabout(self,v):
        """ Reflect ourselves about vector v """
        if issubclass(v.__class__,Vector):
            data = v.data
        else:
            data = np.atleast_1d(v).astype(float)
        # We can use the dot product to project ourselves
        #  onto v
        dp = np.dot(self.data,data)
        projected_on_v = dp/np.linalg.norm(data)
        parallel_to_v = projected_on_v * data/np.linalg.norm(data)
        # Now we need the vector that is perpendicular to v
        # we can get this by using vector addition/subtraction
        # self = parallel_to_v + perpendicular_to_v
        # perpendicular_to_v = self - parallel_to_v
        perpendicular_to_v = self.data - parallel_to_v
        # To perform the reflection, we have to flip the
        # perpendicular component, just take the negative
        reflected = parallel_to_v - perpendicular_to_v
        return self.__class__(reflected)
        
    @property
    def rotation_matrix(self):
        """ Return the rotation matrix that will rotate you into this frame."""
        # just two rotations
        # 1) phi about z-axis (if phi != 0)
        # 2) theta about new y-axis
        if self.phi != 0:
            rot = utils.rotation(([2,self.phi],[1,self.theta]),degrees=True)
        else:
            rot = utils.rotation([1,self.theta],degrees=True)
        return rot

    def rotate(self,rot,degrees=False):
        """ Rotate the vector by this rotation(s)."""
        if isinstance(rot,np.ndarray):
            rotmat = rot
        elif isinstance(rot,list) or isinstance(rot,tuple):
            rotmat = utils.rotation(rot,degrees=degrees)
        newvec = np.matmul(self.data,rotmat)
        self.data = newvec

    def toframe(self,obj):
        """ Return a version of vector transformed to the frame of the input object."""
        if hasattr(obj,'normal')==False:
            raise ValueError('input object must have a normal')
        newobj = self.copy()
        rot = obj.normal.rotation_matrix
        newobj.rotate(rot)
        return newobj

    def plot(self,start_point=None,ax=None,color=None):
        """ Make a 3-D plot of the vector """
        import matplotlib.pyplot as plt
        if start_point is None:
            start_point = np.zeros(3,float)
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        x0,y0,z0 = start_point
        x,y,z = start_point+self.data
        ax.quiver(x0, y0, z0, x, y, z, arrow_length_ratio=0.1,color=color)
        ax.scatter(x0,y0,z0,color=color,s=20)
        ax.set_xlim(x0,x)
        ax.set_ylim(y0,y)
        ax.set_zlim(z0,z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
    
    def copy(self):
        return Vector(self.data.copy())

    # arithmetic operations
    def _check_value(self,value):
        """ Check that the input value is okay """
        if len(np.atleast_1d(value)) != 1 and len(np.atleast_1d(value)) != 3:
            raise ValueError('Value must have 1 or 3 elements')
            
    def __add__(self, value):
        if issubclass(value.__class__,self.__class__):
            data = self.data + value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            data = self.data + value
        return self.__class__(data)
        
    def __iadd__(self, value):
        if issubclass(value.__class__,self.__class__):
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
        newdata = self.data.copy()
        if issubclass(value.__class__,self.__class__):
            data = self.data
            newdata -= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newdata -= value
        return self.__class__(newdata)
              
    def __isub__(self, value):
        if issubclass(value.__class__,self.__class__):
            self.data -= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data -= value
        return self

    def __rsub__(self, value):
        return self - value        
    
    def __mul__(self, value):
        newdata = self.data.copy()
        if issubclass(value.__class__,self.__class__):
            newdata *= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newdata *= value
        return self.__class__(newdata)
               
    def __imul__(self, value):
        if issubclass(value.__class__,self.__class__):
            self.data *= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data *= value
        return self
    
    def __rmul__(self, value):
        return self * value
               
    def __truediv__(self, value):
        newdata = self.data.copy()
        if issubclass(value.__class__,self.__class__):
            newdata /= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newdata /= value
        return self.__class__(newdata)
      
    def __itruediv__(self, value):
        if issubclass(value.__class__,self.__class__):
            self.data /= value.data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            self.data /= value
        return self

    def __rtruediv__(self, value):
        data = self.data.copy()
        if issubclass(value.__class__,self.__class__):
            newdata = value.data / data
        else:
            self._check_value(value)
            if len(np.atleast_1d(value))==1: value=np.atleast_1d(value)[0]
            newdata = value / data
        return self.__class__(newdata)

    def __lt__(self, b):
        return self.r < b.r
    
    def __le__(self, b):
        return self.r <= b.r

    def __eq__(self, b):
        return (self.__class__==b.__class__) and np.all(self.data == b.data)
    
    def __ne__(self, b):
        return ~self.__eq__(b)
    
    def __ge__(self, b):
        return self.r >= b.r
    
    def __gt__(self, b):
        return self.r > b.r

    
class NormalVector(Vector):
    """ Normal vector."""

    def __init__(self,data):
        if issubclass(data.__class__,Vector):
            data = data.data
        if len(data)!=3:
            raise ValueError('data needs three elements')
        self.data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self,value):
        if len(value)!=3:
            raise ValueError('value needs three elements')
        pos = np.array(value).astype(float)
        # normalize
        r = np.linalg.norm(pos)
        if r == 0:
            raise ValueError('Vector cannot have zero length')
        pos /= r
        self.__data = pos

    @property
    def angles(self):
        """ Return the phi and theta angles """
        return self.phi,self.theta
        
    @classmethod
    def fromangles(cls,phi,theta,degrees=False):
        """ Construct NormalVector from phi/theta angles """
        # phi is measured from positive x-axis
        # theta is measured from positive z-axis
        phirad,thetarad = phi,theta
        if degrees:
            phirad,thetarad = np.deg2rad(phi),np.deg2rad(theta)
        data = [np.cos(phirad)*np.sin(thetarad),
                np.sin(phirad)*np.sin(thetarad),
                np.cos(thetarad)]
        return NormalVector(data)
    
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
        t = np.atleast_1d(t)
        nt = len(t)
        pos = np.zeros((nt,3),float)
        for i in range(nt):
            pos[i,0] = self.point[0] + t[i]*self.slopes[0]
            pos[i,1] = self.point[1] + t[i]*self.slopes[1]
            pos[i,2] = self.point[2] + t[i]*self.slopes[2]
        if nt==1:
            pos = pos.squeeze()
        return pos

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

    def rotate(self,rot,degrees=False):
        """ Rotate the line by this rotation(s)."""
        if isinstance(rot,np.ndarray):
            rotmat = rot
        elif isinstance(rot,list) or isinstance(rot,tuple):
            rotmat = utils.rotation(rot,degrees=degrees)
        newslopes = np.matmul(self.slopes,rotmat)
        self.slopes = newslopes
    
    def toframe(self,obj):
        """ Return a version of line transformed to the frame of the input object."""
        # transform two points along the line and then make a new Line out of those
        pnt1 = Point(self(0)).toframe(obj)
        pnt2 = Point(self(1)).toframe(obj)
        newline = Line.frompoints(pnt1,pnt2)
        return newline
        
    def __repr__(self):
        s = 'Line([(x,y,z)=({:.3f},{:.3f},{:.3f})+t({:.3f},{:.3f},{:.3f}))'.format(*self.point,*self.slopes)
        return s

    def isparallel(self,b):
        """ Figure out if a line is parallel """
        if islinelike(b)==False:
            raise ValueError('b must be a Line-line object')
        data = linelikenormal(b)
        # if cross-product of the two vectors is zero, then they are parallel
        cp = np.cross(self.slopes,data)
        if np.linalg.norm(cp) < EPSILON:
            return True
        else:
            return False

    def isperpendicular(self,b):
        """ Figure out if a line is perpendicular """
        if islinelike(b)==False:
            raise ValueError('b must be a Line-line object')
        data = linelikenormal(b)
        # calculate dot product, zero if they orthogonal
        dotproduct = np.dot(self.slopes,data)
        if np.abs(dotproduct) < EPSILON:
            return True
        else:
            return False
        
    def ison(self,point):
        """ Check if a point is on the line """
        if isinstance(point,Point):
            cen = point.data
        else:
            cen = np.array(point).astype(float)
        # Plug point into line parametric equations and see if there is a solution
        #  x02 = x01 + a*t
        #  y02 = y01 + b*t
        #  z02 = z01 + c*t
        # there should be a t value that satisfies all three equations
        #  t = (x02-x01)/a
        #  if a=0, then we have x02=x01
        t = []
        for i in range(3):
            if self.slopes[i]==0:
                if cen[i] != self.point[i]:
                    return False
            else:
                t.append((cen[i]-self.point[i])/self.slopes[i])
        # if there are multiple t's, then make sure they are the same
        if len(t)==1:
            return True
        elif len(t)==2:
            if np.abs(t[0]-t[1]) < EPSILON:
                return True
            else:
                return False
        elif len(t)==3:
            if np.abs(t[0]-t[1]) < EPSILON and np.abs(t[0]-t[2]) < EPSILON:
                return True
            else:
                return False
            
    def plot(self,t=None,ax=None,color=None):
        """ Make a 3-D plot at input points t."""
        if t is None:
            t = np.arange(100)
        pos = self(t)
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.plot(pos[:,0],pos[:,1],pos[:,2],color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
        
    def copy(self):
        return Line(self.point.copy(),self.slopes.copy())

    def __eq__(self, b):
        # Equality is more complicated because two lines can have different points
        # need to do this carefully
        # Parametric form for 3-D line
        # (x,y,z) = (x0,y0,z0)+t(a,b,c)
        sameclass = (self.__class__==b.__class__)
        isparallel = False
        # Check that they are parallel
        if sameclass:
            isparallel = self.isparallel(b)
        # Check that the points on on the line
        isonline = False
        if sameclass and isparallel:
            # check that point2 is on line1
            isonline = self.ison(b.point)
        return (sameclass and isparallel and isonline)
            
    def __ne__(self, b):
        return ~self.__eq__(b)


class Ray(object):

    """ A ray is part of a line with a clear endpoint in one direction that
    stretches indefinitely in the other. Since one side is infinitely long,
    we cannot measure the length of a ray."""

    def __init__(self,position,normal):
        self.position = Point(position)
        self.normal = NormalVector(normal)
    
    def __call__(self,t):
        """ Return the position at parametric value t """
        t = np.atleast_1d(t)
        nt = len(t)
        pos = np.zeros((nt,3),float)
        for i in range(nt):
            if t[i]<0:
                raise ValueError('Only non-negative values allowed')
            pos[i,0] = self.position.x + t[i]*self.normal.data[0]
            pos[i,1] = self.position.y + t[i]*self.normal.data[1]
            pos[i,2] = self.position.z + t[i]*self.normal.data[2]
        if nt==1:
            pos = pos.squeeze()
        return pos

    @classmethod
    def frompoints(cls,p1,p2):
        """ Use two points to define the ray. """
        # first point will be used as the starting point
        pnt1 = np.atleast_1d(p1).astype(float)
        pnt2 = np.atleast_1d(p2).astype(float)
        point = pnt1
        slopes = (pnt2-pnt1)
        return Ray(point,slopes)
    
    @property
    def data(self):
        return self.position.data,self.normal.data

    def __array__(self):
        return self.data

    # def rotate(self,rot,degrees=False):
    #     """ Rotate the ray by this rotation(s)."""
    #     if isinstance(rot,np.ndarray):
    #         rotmat = rot
    #     elif isinstance(rot,list) or isinstance(rot,tuple):
    #         rotmat = utils.rotation(rot,degrees=degrees)
    #     newslopes = np.matmul(self.slopes,rotmat)
    #     self.slopes = newslopes
    
    def toframe(self,obj):
        """ Return a version of line transformed to the frame of the input object."""
        # transform two points along the line and then make a new Line out of those
        pnt1 = self.position.toframe(obj)
        pnt2 = Point(self(1)).toframe(obj)
        newline = Ray.frompoints(pnt1,pnt2)
        return newline
        
    def __repr__(self):
        dd = (*self.position.data,*self.normal.data)
        s = 'Ray([(x,y,z)=({:.3f},{:.3f},{:.3f})+t({:.3f},{:.3f},{:.3f}))'.format(*dd)
        return s

    def isparallel(self,b):
        """ Figure out if a line is parallel """
        if islinelike(b)==False:
            raise ValueError('b must be a Line-line object')
        data = linelikenormal(b)
        # if cross-product of the two vectors is zero, then they are parallel
        cp = np.cross(self.normal.data,data)
        if np.linalg.norm(cp) < EPSILON:
            return True
        else:
            return False

    def isperpendicular(self,b):
        """ Figure out if a line is perpendicular """
        if islinelike(b)==False:
            raise ValueError('b must be a Line-line object')
        data = linelikenormal(b)
        # calculate dot product, zero if they orthogonal
        dotproduct = np.dot(self.normal.data,data)
        if np.abs(dotproduct) < EPSILON:
            return True
        else:
            return False
        
    def ison(self,point):
        """ Check if a point is on the ray """
        if isinstance(point,Point):
            cen = point.data
        else:
            cen = np.array(point).astype(float)
        # Plug point into line parametric equations and see if there is a solution
        #  x02 = x01 + a*t
        #  y02 = y01 + b*t
        #  z02 = z01 + c*t
        # there should be a t value that satisfies all three equations
        #  t = (x02-x01)/a
        #  if a=0, then we have x02=x01
        t = []
        for i in range(3):
            if self.normal.data[i]==0:
                if cen[i] != self.position.data[i]:
                    return False
            else:
                t.append((cen[i]-self.position.data[i])/self.normal.data[i])
        # If there are multiple t's, then make sure they are the same
        # and t >= 0:
        if len(t)==0:
            return False
        elif len(t)==1 and t[0]>=0:
            return True
        elif len(t)==2 and t[0]>=0 and np.abs(t[0]-t[1]) < EPSILON:
            return True
        elif len(t)==3 and t[0]>=0 and np.abs(t[0]-t[1]) < EPSILON and np.abs(t[0]-t[2]) < EPSILON:
            return True
        else:
            return False
            
    def plot(self,ax=None,color=None):
        """ Make a 3-D plot """
        pos = self(t)
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        x0,y0,z0 = self.position.data
        x,y,z = self(5)
        ax.quiver(x0, y0, z0, x, y, z, arrow_length_ratio=0.1,color=color)
        ax.scatter(x0,y0,z0,color=color,s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
        
    def copy(self):
        return Ray(self.position.copy(),self.normal.copy())

    def __eq__(self, b):
        if (self.__class__==b.__class__ and
            np.sum(np.abs(self.position.data-b.position.data)) < EPSILON and
            np.sum(np.abs(self.normal.data-b.normal.data)) < EPSILON):
            return True
        else:
            return False
            
    def __ne__(self, b):
        return ~self.__eq__(b)
    
# square  
# circle
# ellipse
