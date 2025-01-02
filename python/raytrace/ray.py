import os
import numpy as np
from . import surface

# class for light rays

class Ray(object):

    def __init__(self,position,normal):
        self.position = surface.Point(position)
        self.normal = surface.Normal(normal)
        self.history = []
        
    # Needs to keep track of the past points and normals
