# Decorators for raytrace functions and methods

import os
import numpy as np

def arrayize(argument):
    def decorator(function):
        def wrapper(*args, **kwargs):
            funny_stuff()
            something_with_argument(argument)
            result = function(*args, **kwargs)
            more_funny_stuff()
            return result
        return wrapper
    return decorator

# def decorator_factory(argument):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             funny_stuff()
#             something_with_argument(argument)
#             result = function(*args, **kwargs)
#             more_funny_stuff()
#             return result
#         return wrapper
#     return decorator

# class decoratorWithArguments(object):

#     def __init__(self, arg1, arg2, arg3):
#         """
#         If there are decorator arguments, the function
#         to be decorated is not passed to the constructor!
#         """
#         print "Inside __init__()"
#         self.arg1 = arg1
#         self.arg2 = arg2
#         self.arg3 = arg3

#     def __call__(self, f):
#         """
#         If there are decorator arguments, __call__() is only called
#         once, as part of the decoration process! You can only give
#         it a single argument, which is the function object.
#         """
#         print "Inside __call__()"
#         def wrapped_f(*args):
#             print "Inside wrapped_f()"
#             print "Decorator arguments:", self.arg1, self.arg2, self.arg3
#             f(*args)
#             print "After f(*args)"
#         return wrapped_f

class arrayize(object):

    def __init__(self, arg):
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!
        """
        #print("Inside __init__()")
        self.arg = arg

    def __call__(self, fung):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """
        #print("Inside __call__()")
        if self.arg == 'line':
            def wrapper(*args):
                if utils.islinelike(line)==False:
                    raise ValueError('b must be a Line-like object')
                data = utils.linelikenormal(line)
                ndata = data.shape[0]
                out = ndata*[None]
                # Loop over the lines
                for i in range(ndata):
                    out[i] = func(*args)
                return out
            return wrapper
        elif self.arg == 'point':
            def wrapper(*args):
                if utils.ispointlike(line)==False:
                    raise ValueError('b must be a Point-like object')
                data = utils.pointlikedata(line)
                ndata = data.shape[0]
                out = ndata*[None]
                # Loop over the points
                for i in range(ndata):
                    out[i] = func(*args)
                return out
            return wrapper
        else:
            raise ValueError('arrayize decorator type '+str(self.arg)+' not supported')

            
def lineize(func):
    """
    Decorator to handle arrays of lines input to
    a method that handles single lines
    """
    @wraps(func)
    def wrapper(obj1,obj2):
        shape1 = getshape(obj1)
        shape2 = getshape(obj2)
        if shape1 is None or shape2 is None:
            raise ValueError('Input shape must not be None')
        if (len(shape1)>0 or len(shape2)>0) and (shape1 != shape2):
            raise ValueError("Incompatible shapes: {} vs. {}".format(shape1, shape2))
        return func(obj1,obj2)
    return wrapper

def pointize(func):
    """
    Decorator to handle arrays of points input to
    a method that handles single point
    """
    @wraps(func)
    def wrapper(obj1,obj2):
        shape1 = getshape(obj1)
        shape2 = getshape(obj2)
        if shape1 is None or shape2 is None:
            raise ValueError('Input shape must not be None')
        if (len(shape1)>0 or len(shape2)>0) and (shape1 != shape2):
            raise ValueError("Incompatible shapes: {} vs. {}".format(shape1, shape2))
        return func(obj1,obj2)
    return wrapper


def compatibleshapes(func):
    """
    Decorator to check that the shapes of two objects are
    compatible for arithmetic operations.
    """
    @wraps(func)
    def wrapper(obj1,obj2):
        shape1 = getshape(obj1)
        shape2 = getshape(obj2)
        if shape1 is None or shape2 is None:
            raise ValueError('Input shape must not be None')
        if (len(shape1)>0 or len(shape2)>0) and (shape1 != shape2):
            raise ValueError("Incompatible shapes: {} vs. {}".format(shape1, shape2))
        return func(obj1,obj2)
    return wrapper

def compatibleunits(func):
    """
    Decorator to check that the units of two objects are
    compatible for arithmetic operations.
    """
    @wraps(func)
    def wrapper(obj1,obj2):
        if hasattr(obj1,'unit') and obj1.unit is not None and obj1.unit != u.Unit():
            if hasattr(obj2,'unit') and obj2.unit is not None and obj2.unit != u.Unit():
                if obj1.unit != obj2.unit:
                    raise u.UnitsError('Incompatible units')
        return func(obj1,obj2)
    return wrapper
