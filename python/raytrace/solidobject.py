# 3D Shapes or objects with multiple surfaces

import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from . import utils,lightray
from .line import Point,Vector,NormalVector,Line,Ray

EPSILON = 1e-10


class SolidObject(object):
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

    def plot(self):
        pass

    
class Box(SolidObject):

    # 3D box with right-angle sides

    def __init__(self,vertices,**kw):
        self.vertices = vertices

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self,value):
        if utils.ispointlike(value)==False:
            raise ValueError('points must be point-like')
        data = utils.pointlikedata(value)
        npts = data.shape[0]
        if npts != 8:
            raise ValueError('Box needs 8 vertices')
        # check that they are right angles
        self.__vertices = data

    def __repr__(self):
        dd = (*self.position.data,self.normal.data)
        s = 'Box(o=[{:.3f},{:.3f},{:.3f}],radius={:.3f})'.format(*dd)
        return s
        
    def ison(self,points):
        pass
        
    def dointersect(self,line):
        pass
        
    def plot(self):
        pass
        

# cylinder
# cone
# 