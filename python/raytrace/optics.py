# optical elements

import os
import numpy as np
from . import surface,utils
from .lightray import LightRay
from .line import Point,Vector,NormalVector,Line,Ray

#valid_optics = [FlatMirror]

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
        # Construct a plane object
        self.plane = surface.Plane.fromnormalcenter(normal,position)
        self.vertices = vertices

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self,value):
        # Check and save the vertices
        if value is None:
            self.__vertices = None
            self.vertices_inframe = None
            return
        vert = np.atleast_2d(value)
        if vert.ndim != 2 or vert.shape[1]!=3:
            raise ValueError('Vertices must have shape of [N,3]')
        # Now check that they lie on the plane
        onplane = [self.plane.ison(v) for v in vert]
        if np.all(onplane)==False:
            raise ValueError('All vertex points must lie on the plane')
        self.__vertices = vert
        # Construct vertices in the reference frame of the mirror
        self.vertices_inframe = self.toframe(vert)

    @classmethod
    def fromvertices(cls,vertices):
        """ Class method to construct the """
        import pdb; pdb.set_trace()
        # Use the handedness of the vertices to figure out the direction
        vert = np.atleast_2d(vertices).astype(float)
        if vert.ndim != 2 or vert.shape[1]!=3:
            raise ValueError('Vertices must have shape of [N,3]')
        # Get position by taking the mean of the vertex values
        position = np.mean(vert,axis=0)
        # To make the normal use the first two vertices to construct
        #  one vector and the next two to construct a second vector
        #  then take the cross-product to get the normal vector
        # If the first two vectors are parallel, then go to the next
        #  two vertices until you find on that isn't parallel.
        v1 = NormalVector(vert[1,:]-vert[0,:])
        v2 = NormalVector(vert[2,:]-vert[1,:])
        normal = v1.cross(v2)
        return FlatMirror(position,normal,vert)
        
    @property
    def nvertices(self):
        return self.vertices.shape[0]
                
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
        # Check that it is coming from the frontside
        #  dot product needs to be negative
        if self.normal.dot(ray.normal) >= 0:
            # Change the status of the ray
            ray.ray = Ray(position=tpnt,normal=ray.normal)
            ray.state = 'absorbed'
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

        # NEEDS to be able to handle rays hitting the "back" surface
        # they should just be absorbed.
        
    def distance(self,points):
        """ Return distance to plane """
        return self.plane.distance(points)
        
    def toframe(self,points):
        if isinstance(points,Point):
            data = np.atleast_2d(points.data)
        elif (isinstance(points,list) or isinstance(points,tuple)) and isinstance(points[0],Point):
            data = np.atleast_2d([p.data for p in points])
        else:
            data = np.atleast_2d(points)
        npts = data.shape[0]
        newdata = [Point(d).toframe(self).data for d in data]
        newdata = np.atleast_2d(newdata)
        if npts==1:
            newdata = newdata.squeeze()
        return newdata
        
    def ison(self,points):
        """ Check if a point is on the mirror and within the vertices """
        if isinstance(points,Point):
            data = np.atleast_2d(points.data)
        elif (isinstance(points,list) or isinstance(points,tuple)) and isinstance(points[0],Point):
            data = np.atleast_2d([p.data for p in points])
        else:
            data = np.atleast_2d(points)
        npts = data.shape[0]
        isonmirror = npts*[None]
        for i in range(npts):
            # Check if it is on the plane
            isonplane = self.plane.ison(data[i])
            if isonplane==False:
                isonmirror[i] = False
                continue
            # Check if it is inside the vertices
            #  rotate the points into the frame of the plane so it is a 2D problem
            if self.vertices is None:
                isonmirror[i] = True
                continue
            rvert = self.toframe(data[i])
            # isPointInPolygon(xPolygon, yPolygon, xPt, yPt)
            xpoly = self.vertices_inframe[:,0]
            ypoly = self.vertices_inframe[:,1]
            # most edges are not counted as inside
            isonmirror[i] = utils.isPointInPolygon(xpoly,ypoly,rvert[0],rvert[1])
        if len(isonmirror)==1:
            isonmirror = isonmirror[0]
        return isonmirror
            
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
        # Now make sure it's within the vertices
        if self.ison(tpnt)==False:
            return []
        # Make sure that the intersection point is along the direction of the ray
        if ray.ison(tpnt)==False:
            return []
        return tpnt
        
    def plot(self,ax=None,color=None,alpha=0.6):
        """ Make a 3-D plot  """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')

        if self.vertices is not None:
            #from mpl_toolkits.mplot3d import Axes3D
            #from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #ax = Axes3D(fig, auto_add_to_figure=False)
            #fig.add_axes(ax)
            #x = [0,1,1,0]
            #y = [0,0,1,1]
            #z = [0,1,0,1]
            x = self.vertices[:,0]
            y = self.vertices[:,1]
            z = self.vertices[:,2]
            verts = [list(zip(x,y,z))]
            ax.add_collection3d(Poly3DCollection(verts,lw=1,alpha=alpha))
            ax.scatter(x,y,z,color=color)
        else:
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
        
