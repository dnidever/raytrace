# Optical surface

import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation
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
        """ Return the polar angle (angle from positive z-axis) in degrees. """
        return np.rad2deg(np.arctan2(self.rho,self.data[2]))

    @property
    def phi(self):
        """ Return the azimuthal angle (angle from positive x-axis) in degrees."""
        return np.rad2deg(np.arctan2(self.data[1],self.data[0]))

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

    @data.setter
    def data(self,value):
        if len(value)!=3:
            raise ValueError('value needs three elements')
        pos = np.array(value).astype(float)
        self.__data = pos
        
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
        newline = self.copy()
        # translation
        if hasattr(obj,'center')==False:
            raise ValueError('input object must have a center')
        newline.point -= obj.center
        # rotation
        if hasattr(obj,'normal')==False:
            raise ValueError('input object must have a normal')
        rot = obj.normal.rotation_matrix
        newslopes = np.matmul(newline.slopes,rot)
        newline.slopes = newslopes
        return newline
        
    def __repr__(self):
        s = 'Line([(x,y,z)=({:.3f},{:.3f},{:.3f})+t({:.3f},{:.3f},{:.3f}))'.format(*self.point,*self.slopes)
        return s

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
        return self.position.data
    
    def distance(self,obj):
        """ Return distance of point/object to the center of the surface."""
        pass

    def toframe(self,obj):
        """ Return a copy of ourselves transformed into the frame of the input object """
        # translation
        newobj = self.copy()
        newobj.center -= obj.center
        # rotation
        rot = utils.rotation(([2,self.normal.phi],[1,self.normal.theta]),degrees=True)
        
        
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

    def plot(self):
        # Axes3D.plot_wireframe(X, Y, Z, *args, **kwargs)
        # Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
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
        return dt

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
    
    @property
    def equation(self):
        s = '{:.3f}*x + {:.3f}*y + {:.3f}*z + {:.3f} = 0'.format(*self.data)
        return s

    def __repr__(self):
        s = 'Plane('+self.equation+')'
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
    

class Sphere(Surface):

    def __init__(self,radius,**kw):
        super().__init__(**kw)
        self.radius = radius

    def __repr__(self):
        dd = (*self.position.data,*self.normal.data,self.radius)
        s = 'Sphere(o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}],radius={:.3f})'.format(*dd)
        return s
        
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

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot at input points t."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(self.radius*np.cos(u), self.radius*np.sin(v))+self.position.x
        y = np.outer(self.radius*np.sin(u), self.radius*np.sin(v))+self.position.y
        z = np.outer(self.radius*np.ones(np.size(u)), self.radius*np.cos(v))+self.position.z
        # Plot the sphere
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha,
                        cmap=cmap, edgecolors='k', lw=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

    
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

    def __repr__(self):
        dd = (*self.position.data,*self.normal.data,self.radius)
        s = 'HalfSphere(o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}],radius={:.3f})'.format(*dd)
        return s
    
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

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi/2, 100)
        x = np.outer(self.radius*np.cos(u), self.radius*np.sin(v))
        y = np.outer(self.radius*np.sin(u), self.radius*np.sin(v))
        z = np.outer(self.radius*np.ones(np.size(u)), self.radius*np.cos(v))
        # Rotate
        pos = np.zeros((3,100*100),float)
        pos[0,:] = x.ravel()
        pos[1,:] = y.ravel()
        pos[2,:] = z.ravel()
        pos = np.matmul(self.normal.rotation_matrix,pos)
        # translate
        x = pos[0,:] + self.position.x
        x = x.reshape(100,100)
        y = pos[1,:] + self.position.y
        y = y.reshape(100,100)
        z = pos[2,:] + self.position.z
        z = z.reshape(100,100)        
        # Plot the sphere
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha,
                        cmap=cmap, edgecolors='k', lw=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
    
    
class Parabola(Surface):

    def __init__(self,a,position=None,normal=None):
        # a is the leading coefficient of the parabola and determines the shape
        # negative a is convex
        # positive a is concave
        self.__a = float(a)
        if self.__a == 0:
            raise ValueError('a must be nonzero')
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)
        if normal is None:
            normal = [0.0,0.0,1.0]
        self.normal = NormalVector(normal)

        # parabola equation in 3D
        # z = a*(x**2 + y**2)

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self,value):
        self.__a = a

    def __repr__(self):
        dd = (*self.position.data,*self.normal.data,self.a)
        s = 'Parabola(o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}],a={:.3f})'.format(*dd)
        return s

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
        return Line(self.center.copy(),self.normal.data.copy())
    
    @property
    def directrix(self):
        """ Return the directrix. """
        pass

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Create a grid of points, circular region
        phi = np.linspace(0,2*np.pi,50)
        rr = np.linspace(0,5,50)
        X = np.outer(rr,np.cos(phi))
        Y = np.outer(rr,np.sin(phi))
        #xarr = np.linspace(-5, 5, 50)
        #yarr = np.linspace(-5, 5, 50)
        #X, Y = np.meshgrid(xarr, yarr)
        Z = self.a*(X**2+Y**2)
        # Rotate
        pos = np.zeros((3,50*50),float)
        pos[0,:] = X.ravel()
        pos[1,:] = Y.ravel()
        pos[2,:] = Z.ravel()
        pos = np.matmul(self.normal.rotation_matrix,pos)
        # translate
        x = pos[0,:] + self.position.x
        x = x.reshape(50,50)
        y = pos[1,:] + self.position.y
        y = y.reshape(50,50)
        z = pos[2,:] + self.position.z
        z = z.reshape(50,50)        
        # Plot the sphere
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha,
                        cmap=cmap, edgecolors='k', lw=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
