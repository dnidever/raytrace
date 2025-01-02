import os
import numpy as np

# class for light rays

class Ray(object):

    def __init__(self,position,normal):
        self.position = position
        self.normal = normal

    # Needs to keep track of the past points and normals
