# Utility functions for raytrace

import os
import numpy as np

def rotation_matrix(axis,angle,degrees=False):
    """ 3x3 rotation matrix around a single axis """
    anglerad = angle
    if degrees:
        anglerad = np.deg2rad(angle)
    c = np.cos(anglerad)
    s = np.sin(anglerad)
    rot = np.zeros((3,3),float)
    if axis==0:   # x-axis
        rot[0,:] = [ 1, 0, 0]
        rot[1,:] = [ 0, c,-s]
        rot[2,:] = [ 0, s, c]
    elif axis==1: # y-axis
        rot[0,:] = [ c, 0, s]
        rot[1,:] = [ 0, 1, 0]
        rot[2,:] = [-s, 0, c]
    elif axis==2: # z-axis
        rot[0,:] = [ c,-s, 0]
        rot[1,:] = [ s, c, 0]
        rot[2,:] = [ 0, 0, 1]
    else:
        raise ValueError('Only axis=0,1,2 supported')
    return rot

def rotation(values,degrees=False):
    """ Create rotation matrix from multiple rotations."""
    # input is a list/tuple of rotations
    # each rotation is a 2-element list/tuple of (axis,angle)

    # Single rotation
    if isinstance(values[0],list)==False and isinstance(values[0],tuple)==False:
        values = [values]

    # Rotation loop
    rot = np.identity(3)
    for i in range(len(values)):
        axis,angle = values[i]
        rr = rotation_matrix(axis,angle,degrees=degrees)
        rot = np.matmul(rot,rr)
    return rot

    
# intersection function
def intersect_line_plane(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = p1-p0
    dot = p_no*u
    #u = sub_v3v3(p1, p0)
    #dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0-p_co
        fac = -(p_no-w0)/dot
        u *= fac
        out = p0+u
        return out
        #w = sub_v3v3(p0, p_co)
        #fac = -dot_v3v3(p_no, w) / dot
        #u = mul_v3_fl(u, fac)
        #return add_v3v3(p0, u)
    
    # The segment is parallel to plane.
    return None


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )

def intersect_line_plane(line,plane):
    # line in parametric form
    # substitute x/y/z in parabola equation for the line parametric equations
    # solve quadratic equation in t

    # line
    #  x = x0+sx*t
    #  y = y0+sb*t
    #  z = z0+sz*t
    # plane
    #  a*x + b*y + c*z + d = 0

    #  a(x0+sx*t) + b(y0+sy*t) + c(z0+sz*t) + d = 0
    #  (a*x0+b*y0+c*z0+d) + (a*sx+b*sy+c*sz)*t = 0
    #  t = -(a*x0+b*y0+c*z0+d) / (a*sx+b*sy+c*sz)

    # three options
    # 1) one intersection
    # 2) no intersection (parallel and offset)
    # 3) many intersections (contained in the plane)

    x0,y0,z0 = line.point
    sx,sy,sz = line.slopes
    a,b,c,d = plane.data
    numer = -(a*x0 + b*y0 + c*z0 + d)
    denom = a*sz + b*sy + c*sz

    # 1 intersection
    if denom != 0.0:
        t = numer/denom
        out = line(t)
    # no intersection
    elif denom == 0.0 and numer != 0.0:
        out = None
    # in the plane
    elif denom == 0.0 and numer == 0.0:
        out = np.inf

    # Example of a line in the plane
    # https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/1%3A_Vectors_in_Space/Intersection_of_a_Line_and_a_Plane
    # line
    #  x = 1 + 2t
    #  y = -2 + 3t
    #  z = -1 + 4t
    # plane
    #  x + 2y - 2z = -1
    # (1+2t)+2(−2+3t)−2(−1+4t)=-1
    # (1-4+2+1) + (2+6-8)*t = 0
    # (0) + (0)*t = 0

    return out


def intersect_line_parabola(line,parabola):
    # rotate the line into the Parabola reference frame
    # line in parametric form
    # substitute x/y/z in parabola equation for the line parametric equations
    # solve quadratic equation in t

    # line
    #  x = x0+sx*t
    #  y = y0+sb*t
    #  z = z0+sz*t
    # parabola
    #  z = a*(x**2+y**2)

    #  z0+sz*t = d*(x0+sx*t)^2 + a*(y0+sy*t)^2
    #  z0/a + sz/d*t = x0^2+2*x0*sx*t+sx^2*t^2 + y0^2+2*y0*sy*t+sy^2*t^2
    #  0 = (sx^2+sy^2)*t^2 + (2*x0*sx+2*y0*sy-sz/a)*t + (x0^2+y0^2-z0/a)
    #  quadratic equation in t
    #  t = -B +/- sqrt(B^2-4AC)
    #      --------------------
    #            2A

    # A = sx^2+sy^2
    # B = 2*x0*sx+2*y0*sy-sz/a
    # C = x0^2+y0^2-z0/a
    
    # Three options, based on the discriminant
    # 1) B^2-4AC > 0 : 2 solutions
    # 2) B^2-4AC = 0 : 1 solution
    # 3) B^2-4AC < 0 : 0 solutions

    a = parabola.a
    A = line.slopes[0]**2 + line.slopes[1]**2
    B = 2*line.point[0]*line.slopes[0] + 2*line.point[1]*line.slopes[1] - line.slopes[2]/a
    C = line.point[0]**2 + line.point[1]**2 - line.point[2]/a
    discriminant = B**2-4*A*C

    # 2 solutions
    if discriminant > 0:
        t1 = (-B-np.sqrt(discriminant))/(2*A)
        t2 = (-B+np.sqrt(discriminant))/(2*A)
        t = [t1,t2]
        if t2<t1:
            t = t[t2,t1]
    # 1 solution
    elif discriminant == 0:
        t = [-B/(2*A)]
    # 0 solutions
    elif discriminant < 0:
        t = []

    # Get the points
    #   just plug t into the line
    out = []
    for i in range(len(t)):
        pt = line(t[i])
        out.append(pt)

    return out


def normal_to_rot_matrix(normal_vector, angle):
    """ Convert normal vector to a rotation matrix using Rodrigues' formula"""
    n = normal_vector / np.linalg.norm(normal_vector)

    skew_n = np.array([ [0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0] ])

    rot_matrix = np.eye(3) + np.sin(angle) * skew_n + (1 - np.cos(angle)) * np.outer(n, n)

    return rot_matrix

# Example usage

#normal_vec = np.array([1, 0, 0])  # Rotation around X-axis

#rotation_angle = np.radians(90)  # 90 degrees

#rotation_matrix = normal_to_rot_matrix(normal_vec, rotation_angle)

#print(rotation_matrix) 


def euler_rot_matrix(alpha,beta,gamma,degrees=False):
    # Input the euler angles
    # https://en.wikipedia.org/wiki/Euler_angles

    # Z_alpha X_beta Z_gamma convention
    if degrees:
        alpharad = np.deg2rad(alpha)
        ca,sa = np.cos(alpharad),np.sin(alpharad)
        betarad = np.deg2rad(beta)
        cb,sb = np.cos(betarad),np.sin(betarad)
        gammarad = np.deg2rad(gamma)
        cg,sg = np.cos(gammarad),np.sin(gammarad)
    else:
        ca,sa = np.cos(alpha),np.sin(alpha)
        cb,sb = np.cos(beta),np.sin(beta)
        cg,sg = np.cos(gamma),np.sin(gamma)
    rot = np.zeros((3,3),float)
    rot[0,:] = [ca*cg-cb*sa*sg, -ca*sg-cb*cg*sa, sa*sb]
    rot[1,:] = [cg*sa+ca*cb*sg, ca*cb*cg-sa*sg, -ca*sb]
    rot[2,:] = [sb*sg, cg*sb, cb]
    return rot

def euler_angles_from_rot(rot,degrees=False):
    # Z_alpha X_beta Z_gamma convention
    alpha = np.arctan2(rot[0,2],-rot[1,2])   # arctan(R13/-R23)
    beta = np.arccos(rot[2,2])               # arccos(R33)
    gamma = np.arctan2(rot[2,0],rot[2,1])    # arctan(R31/R32)
    if degrees:
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
        gamma = np.rad2deg(gamma)
    return alpha,beta,gamma
