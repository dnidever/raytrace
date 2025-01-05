import os
import numpy as np
from . import surface,line

# class for light rays

cspeed = 2.99792458e8  # speed of light in m/s
hplanck = 6.626e-34    # planck's constant, in J*s

lightray_states = set(['inflight','detected'])

class LightRay(object):

    # Needs to keep track of the past points and normals
    
    def __init__(self,wavelength,position,normal):
        # wavelength in meters
        self.wavelength = wavelength
        self.ray = line.Ray(position,normal)
        self.__path = []
        self.state = 'inflight'
        self.addpath(position)   # add current position to the path
        
    @property
    def wavelength(self):
        return self.__wavelength

    @wavelength.setter
    def wavelength(self,value):
        if value <= 0.0:
            raise ValueError('Wavelength must be non-negative')
        self.__wavelength = float(value)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self,value):
        if value not in lightray_states:
            raise ValueError('Not a recognized LightRay state')
        self.__state = state

    def addtopath(self,point):
        """ Add a point to the path """
        if isinstance(point,line.Point):
            data = point.data.copy()
        else:
            data = np.atleast_1d(point).astype(float)
            if len(data) != 3:
                raise ValueError('Point must have 3 elements')
        self.path.append(data)

    @property
    def path(self):
        """ Return the path """
        return self.__path
        
    def __repr__(self):
        dd = (self.wavelength,*self.position.data,*self.normal.data)
        s = 'Ray(wave={:.3e},p=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
        return s
        
    @property
    def frequency(self):
        # c = wave*frequency
        # frequency in hertz
        return cspeed/self.wavelength

    @property
    def energy(self):
        # E = h*f
        return hplanck*self.frequency
    
    def plot(self,ax=None,color=None):
        """ Make a 3-D plot of the ray """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        coords = np.atleast_2d(self.path)
        npnts = coords.shape[0]
        # Always draw lines between all the path points (if more than 1)
        if npnts>1:
            ax.plot(coords[:,0],coords[:,1],coords[:,2],color=color)
            # Add point for initial position
            ax.scatter(coords[0,0],coords[0,1],coords[0,2],s=20,color='r')
        # Add point for current position
        ax.scatter(coords[-1,0],coords[-1,1],coords[-1,2],s=20,color='green')
        # If it is still 'inflight', then add arrow at the end
        if self.state=='inflight':
            x0,y0,z0 = coords[-1,:]
            x = x0 + self.normal.data[0]*3
            y = y0 + self.normal.data[1]*3
            z = z0 + self.normal.data[2]*3
            ax.quiver(x0, y0, z0, x, y, z, arrow_length_ratio=0.1,color=color)
        xr = [np.min(coords[:,0]),np.max(coords[:,0])]
        dx = np.maximum(np.ptp(xr),1)
        xr = [xr[0]-0.1*dx,xr[1]+0.1*dx]
        yr = [np.min(coords[:,1]),np.max(coords[:,1])]
        dy = np.maximum(np.ptp(yr),1)
        yr = [yr[0]-0.1*dy,yr[1]+0.1*dy]
        zr = [np.min(coords[:,2]),np.max(coords[:,2])]
        dz = np.maximum(np.ptp(zr),1)
        zr = [zr[0]-0.1*dz,zr[1]+0.1*dz]
        ax.set_xlim(xr)
        ax.set_ylim(yr)
        ax.set_zlim(zr)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
