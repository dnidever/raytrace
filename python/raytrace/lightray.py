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
        self.__history = []
        # wavelength in meters
        self.wavelength = wavelength
        self.ray = line.Ray(position,normal)
        self.state = 'inflight'
        
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
        self.__state = value

    @property
    def ray(self):
        return self.__ray

    @ray.setter
    def ray(self,value):
        if isinstance(value,line.Ray)==False:
            raise ValueError('Must be Ray')
        self.__ray = value
        # Save both the position and normal values
        #  the setter will append it to the history
        self.history = (value.position.data.copy(),value.normal.data.copy())
        
    @property
    def history(self):
        return self.__history

    @history.setter
    def history(self,value):
        self.__history.append(value)
    
    @property
    def path(self):
        """ Return the path """
        return np.atleast_2d([r[0] for r in self.history])

    # def addtopath(self,point):
    #     """ Add a point to the path """
    #     if isinstance(point,line.Point):
    #         data = point.data.copy()
    #     else:
    #         data = np.atleast_1d(point).astype(float)
    #         if len(data) != 3:
    #             raise ValueError('Point must have 3 elements')
    #     self.path.append(data)

    @property
    def position(self):
        return self.ray.position

    @property
    def normal(self):
        return self.ray.normal
        

        
    def __repr__(self):
        dd = (self.state,self.wavelength,*self.position.data,*self.normal.data)
        s = 'LightRay({:},wave={:.3e},o=[{:.3f},{:.3f},{:.3f}],n=[{:.3f},{:.3f},{:.3f}])'.format(*dd)
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
    
    def plot(self,ax=None,color=None,alpha=0.8):
        """ Make a 3-D plot of the ray """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        coords = np.atleast_2d(self.path)
        npnts = coords.shape[0]
        # Always draw lines between all the path points (if more than 1)
        if npnts>1:
            ax.plot(coords[:,0],coords[:,1],coords[:,2],color=color,alpha=alpha)
            # Add point for initial position
            ax.scatter(coords[0,0],coords[0,1],coords[0,2],s=20,color='r',alpha=alpha)
        # Add point for current position
        ax.scatter(coords[-1,0],coords[-1,1],coords[-1,2],s=20,color='green',alpha=alpha)
        # If it is still 'inflight', then add arrow at the end
        if self.state=='inflight':
            x,y,z = coords[-1,:]
            u,v,w = self.normal.data
            # x/y/z is the position of the arrow (by default the tail)
            # u/v/w are the components of the arrow vectors
            ax.quiver(x, y, z, u, v, w, color='k',linewidth=2,alpha=alpha) #arrow_length_ratio=0.1,color=color)
        xr = [np.min(coords[:,0]),np.max(coords[:,0])]
        dx = np.maximum(np.ptp(xr)*0.1,1)
        xr = [xr[0]-dx,xr[1]+dx]
        yr = [np.min(coords[:,1]),np.max(coords[:,1])]
        dy = np.maximum(np.ptp(yr)*0.1,1)
        yr = [yr[0]-dy,yr[1]+dy]
        zr = [np.min(coords[:,2]),np.max(coords[:,2])]
        dz = np.maximum(np.ptp(zr)*0.1,1)
        zr = [zr[0]-dz,zr[1]+dz]
        xr0,yr0,zr0 = ax.get_xlim(),ax.get_ylim(),ax.get_zlim()
        fxr = [min(xr[0],xr0[0]),max(xr[1],xr0[1])]
        fyr = [min(yr[0],yr0[0]),max(yr[1],yr0[1])]
        fzr = [min(zr[0],zr0[0]),max(zr[1],zr0[1])]
        ax.set_xlim(fxr)
        ax.set_ylim(fyr)
        ax.set_zlim(fzr)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
