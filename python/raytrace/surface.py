# Surfaces

import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from . import utils,lightray
from .line import Point,Vector,NormalVector,Line,Ray

EPSILON = 1e-10


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

    def normalatpoint(self,point):
        """ Return the normal at a certain point """
        pass
    
    def toframe(self,obj):
        """ Return a copy of ourselves transformed into the frame of the input object """
        # translation
        newobj = self.copy()
        newobj.center -= obj.center
        # rotation
        rot = utils.rotation(([2,self.normal.phi],[1,self.normal.theta]),degrees=True)
        
    def dointersect(self,line):
        """ Does the line intersect the surface """
        intpts = self.intersections(line)
        if len(intpts)==0:
            return False
        else:
            return True

    def intersections(self,line):
        """ Return the intersection points """
        pass

    def plot(self):
        """ Make 3D plot of the surface """
        pass
        
    def copy(self):
        return copy.deepcopy(self)
    

class Plane(Surface):

    def __init__(self,normal,d):
        # the normal vector normalizes the vector
        # which we don't want for the defintion of the plane
        # because we need to divide "d" as well
        r = np.linalg.norm(normal)
        self.normal = NormalVector(normal)
        self.d = d / r
        self.position = None

        # Equation of a plane in 3D
        # a*x + b*y + c*z + d = 0

        # the normal vector is (a,b,c)

    @property
    def data(self):
        dt = np.zeros(4,float)
        dt[:3] = self.normal.data
        dt[3] = self.d
        return dt

    # def __call__(self,t):
    #     """ Return the position at parametric value t """
    #     t = np.atleast_1d(t)
    #     nt = len(t)
    #     pos = np.zeros((nt,3),float)
    #     for i in range(nt):
    #         pos[i,0] = self.point[0] + t[i]*self.slopes[0]
    #         pos[i,1] = self.point[1] + t[i]*self.slopes[1]
    #         pos[i,2] = self.point[2] + t[i]*self.slopes[2]
    #     if nt==1:
    #         pos = pos.squeeze()
    #     return pos

    @classmethod
    def fromnormalcenter(cls,normal,position):
        """ Class method to construct a Plen from a normal and center """
        if isinstance(normal,NormalVector):
            norm = normal.data.copy()
        else:
            norm = np.atleast_1d(normal).astype(float)
        if isinstance(position,Point):
            cen = position.data.copy()
        else:
            cen = np.atleast_1d(position).astype(float)
        # Equation of a plane in 3D
        # a*x + b*y + c*z + d = 0
        # the normal vector is (a,b,c)
        # to solve for d, put the x/y/z center values in the equation and solve for d
        # d = -(a*x0+b*y0+c*z0)
        d = -np.sum(norm*cen)
        return Plane(normal,d)
        
    @property
    def equation(self):
        s = '{:.3f}*x + {:.3f}*y + {:.3f}*z + {:.3f} = 0'.format(*self.data)
        return s

    def __repr__(self):
        s = 'Plane('+self.equation+')'
        return s
    
    def distance(self,obj):
        """ Distance from a point and the normal of the plane. """
        if (isinstance(obj,list)==False and isinstance(obj,tuple)==False and
            (isinstance(obj,np.ndarray) and obj.ndim!=2)):
            obj = [obj]
        # Loop over objects/points
        distances = np.zeros(len(obj),float)
        for i in range(len(obj)):
            o = obj[i]
            if hasattr(o,'center'):
                pnt = o.center
            elif isinstance(o,Point):
                pnt = o
            else:
                pnt = Point(o)
            # d = |A*xo + B*yo + C*zo + D |/sqrt(A^2 + B^2 + C^2),
            # where (xo, yo, zo) is the given point and Ax + By + Cz + D = 0 is the
            x0,y0,z0 = pnt.data
            a,b,c,d = self.data
            numer = np.abs(a*x0+b*y0+c*z0+d)
            denom = np.sqrt(a**2+b**2+c**2)
            if denom != 0.0:
                dist = numer / denom
            else:
                dist = np.nan
            distances[i] = dist
        if len(obj)==1:
            distances = distances[0]
        return distances

    def isparallel(self,line):
        """ Check if a line or ray is parallel to the plane """
        # check if the line direction is orthogonal to the plane's normal vector
        # calculate dot product, zero if they orthogonal
        if utils.islinelike(line)==False:
            raise ValueError('b must be a Line-like object')
        data = utils.linelikenormal(line)
        # check if line is perpendicular to plane normal vector
        return self.normal.isperpendicular(data)

    def normalatpoint(self,point):
        """ Return the normal at a certain point """
        return self.normal
    
    def isperpendicular(self,line):
        """ Check if a line of ray is perpendicular to the plane """
        # check if the line normal is orthogonal to the plane's normal vector
        # calculate dot product, zero if they orthogonal
        if utils.islinelike(line)==False:
            raise ValueError('b must be a Line-like object')
        data = utils.linelikenormal(line)
        # check if line is parallel to plane normal vector
        return self.normal.isparallel(data)
        
    def ison(self,points):
        """ Check if points lie on the plane """
        if utils.ispointlike(points)==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(points)
        npts = data.shape[0]
        isonplane = npts*[None]
        for i in range(npts):
            # Plug the point coordinate into the parameteric equations
            # and see if it is true.
            # a*x + b*y + c*z + d = 0
            if np.abs(np.sum(self.normal.data*data[i,:])+self.d) < EPSILON:
                isonplane[i] = True
            else:
                isonplane[i] = False
        if npts==1:
            isonplane = isonplane[0]
        return isonplane
    
    def intersections(self,line):
        """ Return the intersection points """
        if isinstance(line,lightray.LightRay):
            l = Line(line.ray.position.data,line.ray.normal.data)
        elif isinstance(line,Ray):
            l = Line(line.position.data,line.normal.data)
        elif isinstance(line,Line):
            l = line
        else:
            raise ValueError('input must be Line, Ray or LightRay')
        out = utils.intersect_line_plane(l,self)
        return out
        
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

    def __eq__(self, b):
        return ((self.__class__==b.__class__) and np.all(self.data==b.data))
    
    def __ne__(self, b):
        return ((self.__class__!=b.__class__) or np.any(self.data!=b.data))
    

class Sphere(Surface):

    def __init__(self,radius,position=None):
        self.radius = radius
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)

    @property
    def data(self):
        return *self.position.data,self.radius
        
    def __repr__(self):
        dd = (*self.position.data,self.radius)
        s = 'Sphere(o=[{:.3f},{:.3f},{:.3f}],radius={:.3f})'.format(*dd)
        return s

    def normalatpoint(self,point):
        """ Return the normal at a certain point """
        raise NotImplemented
    
    def distance(self,obj):
        if hasattr(obj,center):
            pnt = obj.center
        elif isinstance(obj,Point):
            pnt = obj
        else:
            pnt = Point(obj)
        return np.linalg.norm(self.center-pnt)

        return self.normal

    def ison(self,points):
        """ Check if points lie on the plane """
        if np.all(utils.ispointlike(points))==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(points)
        npts = data.shape[0]
        isonsphere = npts*[None]
        for i in range(npts):
            # Check if the point is on the sphere
            pnt = Point(data[i,:]).toframe(self)
            x0,y0,z0 = pnt.data
            z = self.a*(x0**2+y0**2)
            if np.abs(pnt.r-self.radius) < EPSILON:
                isonsphere[i] = True
            else:
                isonsphere[i] = False
        if npts==1:
            isonsphere = isonsphere[0]
        return isonsphere
    
    def intersections(self,line):
        """ Return the intersection points """
        if isinstance(line,lightray.LightRay):
            l = Line(line.position.data,line.normal.data)
        elif isinstance(line,Line):
            l = line
        else:
            raise ValueError('input must be Line or Ray')
        out = utils.intersect_line_sphere(l,self)
        # Sort by distance if ray/line has a position
        if len(out)>1 and hasattr(l,'position'):
            dist = [l.position.distance(o) for o in out]
            if dist[0] > dist[1]:  # flip the order
                out = [out[1],out[0]]
        return out

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot at input points t."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u),np.sin(v))*self.radius+self.position.x
        y = np.outer(np.sin(u),np.sin(v))*self.radius+self.position.y
        z = np.outer(np.ones(np.size(u)),np.cos(v))*self.radius+self.position.z
        # Plot the sphere
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha,
                        cmap=cmap, edgecolors='k', lw=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

    def __eq__(self, b):
        return ((self.__class__==b.__class__) and np.all(self.data==b.data) and
                 (self.position==b.position))
    
    def __ne__(self, b):
        return ((self.__class__!=b.__class__) or np.any(self.data!=b.data) or
                 (self.position!=b.position))
    
    
class HalfSphere(Surface):

    """ Half Sphere, flat on other (bottom) side """
    
    def __init__(self,radius,position=None,normal=None):
        if position is None:
            position = [0.0,0.0,0.0]
        self.position = Point(position)
        if normal is None:
            normal = [0.0,0.0,1.0]
        self.normal = NormalVector(normal)
        # negative radius is convex
        # positive radius is concave
        self.radius = radius
        
    @property
    def convex(self):
        return self.radius

    @property
    def data(self):
        return *self.position.data,self.radius
    
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

    def normalatpoint(self,point):
        """ Return the normal at a certain point """
        raise NotImplemented
    
    @property
    def bottomplane(self):
        """ Return the plane of the bottom of the half sphere """
        # equation of plane is
        # a*x + b*y + c*z + d = 0
        # where the normal vector is (a,b,c)
        # so we just need to find d
        # put our center point into the equation and solve for d
        # d = -(a*x0+b*y0+c*z0)
        d = -np.sum(self.normal.data*self.center)
        p = Plane(self.normal.data,d)
        p.radius = self.radius
        return p

    def ison(self,points):
        """ Check if points lie on the plane """
        if np.all(utils.ispointlike(points))==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(points)
        npts = data.shape[0]
        isonhsphere = npts*[None]
        for i in range(npts):
            # Check if the point is on the half sphere
            pnt = Point(data[i,:]).toframe(self)
            x0,y0,z0 = pnt.data
            z = self.a*(x0**2+y0**2)
            if np.abs(pnt.r-self.radius) < EPSILON and z0 >= 0.0:
                isonhsphere[i] = True
            else:
                isonhsphere[i] = False
        if npts==1:
            isonhsphere = isonhsphere[0]
        return isonhsphere
    
    def intersections(self,line):
        """ Return the intersection points """
        if isinstance(line,lightray.LightRay):
            l = Line(line.position.data,line.normal.data)
        elif isinstance(line,Line):
            l = line
        else:
            raise ValueError('input must be Line or Ray')
        out = utils.intersect_line_halfsphere(l,self)
        # Sort by distance if ray/line has a position
        if len(out)>1 and hasattr(l,'position'):
            dist = [l.position.distance(o) for o in out]
            if dist[0] > dist[1]:  # flip the order
                out = [out[1],out[0]]
        return out

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi/2, 100)
        x = np.outer(np.cos(u), np.sin(v))*self.radius
        y = np.outer(np.sin(u), np.sin(v))*self.radius
        z = np.outer(np.ones(np.size(u)), np.cos(v))*self.radius
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

    def __eq__(self, b):
        return ((self.__class__==b.__class__) and np.all(self.data==b.data) and
                (self.position==b.position) and (self.normal==b.normal))
    
    def __ne__(self, b):
        return ((self.__class__!=b.__class__) or np.any(self.data!=b.data) or
                (self.position!=b.position) or (self.normal!=b.normal))

    
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
        self.__a = value

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

    def normalat(self,point):
        """ Return the normal at a certain point """
        raise NotImplemented
    
    def ison(self,pnt):
        """ Check if points lie on the plane """
        if np.all(utils.ispointlike(points))==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(points)
        npts = data.shape[0]
        isonparabola = npts*[None]
        for i in range(npts):
            # Check if the point is on the parabola
            pnt = Point(data[i,:]).toframe(self)
            x0,y0,z0 = pnt.data
            z = self.a*(x0**2+y0**2)
            if np.abs(z-z0) < EPSILON:
                isonparabola[i] = True
            else:
                isonparabola[i] = False
        if npts==1:
            isonparabola = isonparabola[0]
        return isonparabola

    def intersections(self,line):
        """ Return the intersection points """
        # rotate the line into the Parabola reference frame
        # line in parametric form
        # substitute x/y/z in parabola equation for the line parametric equations
        # solve quadratic equation in t
        #rline = line.rotate(self.normal)
        #intpts = utils.intersect_line_parabola(line,self)
        #return intpts
        if isinstance(line,lightray.LightRay):
            l = Line(line.position.data,line.normal.data)
        elif isinstance(line,Line):
            l = line
        else:
            raise ValueError('input must be Line or Ray')
        out = utils.intersect_line_parabola(l,self)
        # Sort by distance if ray/line has a position
        if len(out)>1 and hasattr(l,'position'):
            dist = [l.position.distance(o) for o in out]
            if dist[0] > dist[1]:  # flip the order
                out = [out[1],out[0]]
        return out
    
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
        pos = np.zeros((50*50,3),float)
        pos[:,0] = X.ravel()
        pos[:,1] = Y.ravel()
        pos[:,2] = Z.ravel()
        pos = np.matmul(pos,self.normal.rotation_matrix.T)
        # translate
        x = pos[:,0] + self.position.x
        x = x.reshape(50,50)
        y = pos[:,1] + self.position.y
        y = y.reshape(50,50)
        z = pos[:,2] + self.position.z
        z = z.reshape(50,50)        
        # Plot the sphere
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha,
                        cmap=cmap, edgecolors='k', lw=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

    def __eq__(self, b):
        return ((self.__class__==b.__class__) and (self.a==b.a) and
                (self.position==b.position) and (self.normal==b.normal))
    
    def __ne__(self, b):
        return ((self.__class__!=b.__class__) or (self.a!=b.a) or
                (self.position!=b.position) or (self.normal!=b.normal))

    
####--------------------------------
#  Bounded Surfaces

# class BoundedSurface(Surface):
#     """ These are 2D surface that have edges """

#     def __init__(self,boundary,*args,**kw):
#         super().__init__(*args,**kw)
#         self.boundary = boundary


class Rectangle(Surface):

    def __init__(self,boundary,**kw):
        super().__init__(**kw)
        self.boundary = boundary
        d = -np.sum(self.boundary[0,:]*self.normal.data)
        self.plane = Plane(self.normal.data,d)

    @property
    def flat(self):
        return True

    @property
    def boundary(self):
        return self.__boundary

    @boundary.setter
    def boundary(self,value):
        vertices = np.atleast_2d(value)
        if vertices.shape[0] != 4 or vertices.shape[1] != 3:
            raise ValueError('boundary must have 4 points with x/y/z coordinates')
        # check that we have four right angles, dot products should be zero
        v1 = NormalVector(vertices[1,:]-vertices[0,:])
        v2 = NormalVector(vertices[2,:]-vertices[1,:])
        v3 = NormalVector(vertices[3,:]-vertices[2,:])
        v4 = NormalVector(vertices[0,:]-vertices[3,:])
        d12 = v1.dot(v2)
        d23 = v2.dot(v3)
        d34 = v3.dot(v4)
        d41 = v4.dot(v1)
        if np.sum(np.abs([d12,d23,d34,d41])) > 4*EPSILON:
            raise ValueError('A rectangle needs four right angles')
        # check that they all fall on a plane
        #  calculate the normal vector
        norm = v1.cross(v2)
        d = -np.sum(vertices[0,:]*norm.data)
        plane = Plane(norm.data,d)
        onplane = plane.ison(vertices)
        if np.all(onplane)==False:
            raise ValueError('Rectangle vertices must lie in a plane')
        self.__boundary = vertices

    @classmethod
    def fromvertices(cls,value):
        """ Create the object from the vertices themselves """
        # determine the center and normal directly
        vertices = np.atleast_2d(value)
        if vertices.shape[0] != 4 or vertices.shape[1] != 3:
            raise ValueError('boundary must have 4 points with x/y/z coordinates')
        v1 = NormalVector(vertices[1,:]-vertices[0,:])
        v2 = NormalVector(vertices[2,:]-vertices[1,:])
        norm = v1.cross(v2)
        pos = np.mean(vertices,axis=0)
        return cls(vertices,position=pos,normal=norm)

    def __repr__(self):
        dd = (*self.position.data,*self.normal.data)
        s = 'Rectangle(o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
        return s

    def ison(self,points):
        """ Check if points lie on the plane """
        if np.all(utils.ispointlike(points))==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(points)
        npts = data.shape[0]
        vert = np.atleast_2d([Point(p).toframe(self).data for p in self.boundary])
        isonrec = npts*[None]
        for i in range(npts):
            # Check if the point is on the rectangle
            pnt = Point(data[i,:]).toframe(self)
            onplane = self.plane.ison(pnt)
            if self.plane.ison(pnt) and utils.isPointInPolygon(vert[:,0],vert[:,1],pnt.x,pnt.y):
                isonrec[i] = True
            else:
                isonrec[i] = False
        if npts==1:
            isonrec = isonrec[0]
        return isonrec
        
    def intersections(self,line):
        """ Return the intersection points """
        # rotate the line into the Parabola reference frame
        # line in parametric form
        # substitute x/y/z in parabola equation for the line parametric equations
        # solve quadratic equation in t
        #rline = line.rotate(self.normal)
        #intpts = utils.intersect_line_parabola(line,self)
        #return intpts
        if utils.sisinstance(line,'raytrace.lightray.LightRay'):
            l = Line(line.position.data,line.normal.data)
        elif utils.sisinstance(line,'raytrace.line.Line'):
            l = line
        else:
            raise ValueError('input must be Line or Ray')
        out = utils.intersect_line_plane(l,self.plane)
        # Rotate to the rectangle frame and make sure it is within the vertices
        if len(out)>0:
            rout = Point(out).toframe(self).data
            vert = np.atleast_2d([Point(v).toframe(self) for v in self.boundary])
            if utils.isPointInPolygon(vert[:,0],vert[:,1],rout[0],rout[1]):
                return out
            else:
                return []
        else:
            return []

    def plot(self,ax=None,color=None,alpha=0.6,cmap='viridis'):
        """ Make a 3-D plot """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        import pdb; pdb.set_trace()
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi/2, 100)
        x = np.outer(np.cos(u), np.sin(v))*self.radius
        y = np.outer(np.sin(u), np.sin(v))*self.radius
        z = np.outer(np.ones(np.size(u)), np.cos(v))*self.radius
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
