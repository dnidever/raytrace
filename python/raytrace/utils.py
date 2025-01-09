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
    #  y = y0+sy*t
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
    if str(plane.__class__)=="<class 'raytrace.surface.Plane'>":
        a,b,c,d = plane.data
    else:
        a,b,c,d = plane
    numer = -(a*x0 + b*y0 + c*z0 + d)
    denom = a*sx + b*sy + c*sz

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

def intersect_line_sphere(line,sphere):
    # line in parametric form
    # substitute x/y/z in parabola equation for the line parametric equations
    # solve quadratic equation in t

    # line
    #  x = x0+sx*t
    #  y = y0+sy*t
    #  z = z0+sz*t
    # sphere
    #  (x-a)**2 + (y-b)**2 + (z-c)**2 = r**2

    #  (x0+sx*t-a)**2 + (y0+sy*t-b)**2 + (z0+sz*t-c)**2 = r**2
    #
    #  (x0-a)**2 + 2*(x0-a)*sx*t + sx**2*t**2 +
    #  (y0-b)**2 + 2*(y0-b)*sy*t + sy**2*t**2 +
    #  (z0-c)**2 + 2*(z0-c)*sz*t + sz**2*t**2 - r**2 = 0
    #
    #  ((x0-a)**2+(y0-b)**2+(z0-c)**2-r**2) +
    #  2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)*t +
    #  (sx**2+sy**2+sz**2)*t**2 = 0

    #  quadratic equation in t
    #  t = -B +/- sqrt(B^2-4AC)
    #      --------------------
    #            2A

    # A = sx**2+sy**2+sz**2
    # B = 2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)
    # C = (x0-a)**2+(y0-b)**2+(z0-c)**2-r**2
    
    # Three options, based on the discriminant
    # 1) B^2-4AC > 0 : 2 solutions
    # 2) B^2-4AC = 0 : 1 solution
    # 3) B^2-4AC < 0 : 0 solutions

    x0,y0,z0 = line.point
    sx,sy,sz = line.slopes
    a,b,c,r = sphere.data
    A = sx**2+sy**2+sz**2
    B = 2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)
    C = (x0-a)**2+(y0-b)**2+(z0-c)**2-r**2
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


def intersect_line_halfsphere(line,halfsphere):
    # line in parametric form
    # substitute x/y/z in parabola equation for the line parametric equations
    # solve quadratic equation in t

    # line
    #  x = x0+sx*t
    #  y = y0+sy*t
    #  z = z0+sz*t
    # sphere
    #  (x-a)**2 + (y-b)**2 + (z-c)**2 = r**2

    #  (x0+sx*t-a)**2 + (y0+sy*t-b)**2 + (z0+sz*t-c)**2 = r**2
    #
    #  (x0-a)**2 + 2*(x0-a)*sx*t + sx**2*t**2 +
    #  (y0-b)**2 + 2*(y0-b)*sy*t + sy**2*t**2 +
    #  (z0-c)**2 + 2*(z0-c)*sz*t + sz**2*t**2 - r**2 = 0
    #
    #  ((x0-a)**2+(y0-b)**2+(z0-c)**2-r**2) +
    #  2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)*t +
    #  (sx**2+sy**2+sz**2)*t**2 = 0

    #  quadratic equation in t
    #  t = -B +/- sqrt(B^2-4AC)
    #      --------------------
    #            2A

    # A = sx**2+sy**2+sz**2
    # B = 2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)
    # C = (x0-a)**2+(y0-b)**2+(z0-c)**2-r**2
    
    # Three options, based on the discriminant
    # 1) B^2-4AC > 0 : 2 solutions
    # 2) B^2-4AC = 0 : 1 solution
    # 3) B^2-4AC < 0 : 0 solutions

    x0,y0,z0 = line.point
    sx,sy,sz = line.slopes
    a,b,c = halfsphere.center
    r = halfsphere.radius
    A = sx**2+sy**2+sz**2
    B = 2*((x0-a)*sx+(y0-b)*sy+(z0-c)*sz)
    C = (x0-a)**2+(y0-b)**2+(z0-c)**2-r**2
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
    sout = []
    for i in range(len(t)):
        pt = line(t[i])
        sout.append(pt)

    # Make sure to only include intersections above the plane
    #  rotate the points into the frame of the halfsphere
    out = []
    for i in range(len(sout)):
        pnt = sout[i].copy()
        pnt -= halfsphere.center
        rot = halfsphere.normal.rotation_matrix
        pnt2 = np.matmul(pnt,rot)
        if pnt2[2] >= 0:
            out.append(sout[i])
        
    # Check if the line passes through the bottom plane
    bottomplane = halfsphere.bottomplane
    pout = intersect_line_plane(line,bottomplane)
    rad = np.linalg.norm(halfsphere.center-pout)
    if rad < bottomplane.radius:
        out.append(pout)
        
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
    #  z = d*(x**2+y**2)

    #  z0+sz*t = a*((x0+sx*t)^2 + (y0+sy*t)^2)
    #  z0/a + sz*t/a = x0^2+2*x0*sx*t+sx^2*t^2 + y0^2+2*y0*sy*t+sy^2*t^2
    #  (sx^2+sy^2)*t^2 + (2*x0*sx+2*y0*sy-sz/a)*t + (x0^2+y0^2-z0/a) = 0
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

    # Rotate the line into the reference frame of the parabola
    rline = line.toframe(parabola)
    
    a = parabola.a
    A = rline.slopes[0]**2 + rline.slopes[1]**2
    B = 2*rline.point[0]*rline.slopes[0] + 2*rline.point[1]*rline.slopes[1] - rline.slopes[2]/a
    C = rline.point[0]**2 + rline.point[1]**2 - rline.point[2]/a
    discriminant = B**2-4*A*C
    
    # Quadratic term is zero, now just a linear equation
    if np.abs(A)<1e-20:
        # B*t + C = 0
        if B!=0:
            t = [-C/B]
    else:
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
    out1 = []
    for i in range(len(t)):
        pt = rline(t[i])
        out1.append(pt)

    # We need to rotate the points back to the original frame
    out = []
    rot = parabola.normal.rotation_matrix.T
    for i in range(len(out1)):
        pt = out1[i].copy()
        pt2 = np.matmul(pt,rot)
        pt2 += parabola.center
        out.append(pt2)
    
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

def stype(obj):
    """ Return the object type in a string """
    # "<class 'raytrace.lightray.LightRay'>"
    value = str(type(obj))
    value = value[8:-2]
    return value

def sisinstance(obj,styp):
    """ Check if the object is the right type using a string of the class name """
    # need to give the full name, e.g. "raytrace.lightray.LightRay"
    objstype = stype(obj)
    if objstype == styp:
        return True
    else:
        return False

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

def doPolygonsOverlap(xPolygon1, yPolygon1, xPolygon2, yPolygon2):
    """Returns True if two polygons are overlapping."""

    # How to determine if two polygons overlap.
    # If a vertex of one of the polygons is inside the other polygon
    # then they overlap.

    n1 = len(xPolygon1)
    n2 = len(xPolygon2)
    isin = False

    # If ranges don't overlap, then polygons don't overlap
    if rangeoverlap(xPolygon1,xPolygon2)==False or rangeoverlap(yPolygon1,yPolygon2)==False:
        return False

    # Loop through all vertices of second polygon
    for i in range(n2):
        # perform iterative boolean OR
        # if any point is inside the polygon then they overlap
        isin = isin or isPointInPolygon(xPolygon1, yPolygon1, xPolygon2[i], yPolygon2[i])

    # Need to do the reverse as well, not the same
    for i in range(n1):
        isin = isin or isPointInPolygon(xPolygon2, yPolygon2, xPolygon1[i], yPolygon1[i])

    # Two polygons can overlap even if there are no vertices inside each other.
    # Need to check if the line segments overlap
    if isin==False:
        intersect = False
        # Add first vertex to the end
        xp1 = np.append( xPolygon1, xPolygon1[0] )
        yp1 = np.append( yPolygon1, yPolygon1[0] )
        xp2 = np.append( xPolygon2, xPolygon2[0] )
        yp2 = np.append( yPolygon2, yPolygon2[0] )
        for i in range(4):
            for j in range(4):
                intersect = intersect or doLineSegmentsIntersect(xp1[i:i+2],yp1[i:i+2],xp2[j:j+2],yp2[j:j+2])
                if intersect==True:
                    return True
        isin = isin or intersect

    return isin

def isPointInPolygon(xPolygon, yPolygon, xPt, yPt):
    """Returns boolean if a point is inside a polygon of vertices."""

    # How to tell if a point is inside a polygon:
    # Determine the change in angle made by the point and the vertices
    # of the polygon.  Add up the delta(angle)'s from the first (include
    # the first point again at the end).  If the point is inside the
    # polygon, then the total angle will be +/-360 deg.  If the point is
    # outside, then the total angle will be 0 deg.  Points on the edge will
    # outside.
    # This is called the Winding Algorithm
    # http://geomalgorithms.com/a03-_inclusion.html

    n = len(xPolygon)
    # Array for the angles
    angle = np.zeros(n)

    # add first vertex to the end
    xPolygon1 = np.append( xPolygon, xPolygon[0] )
    yPolygon1 = np.append( yPolygon, yPolygon[0] )

    wn = 0   # winding number counter

    # Loop through the edges of the polygon
    for i in range(n):
        # if edge crosses upward (includes its starting endpoint, and excludes its final endpoint)
        if yPolygon1[i] <= yPt and yPolygon1[i+1] > yPt:
            # if (P is  strictly left of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) > 0:
                 wn += 1   # a valid up intersect right of P.x

        # if edge crosses downward (excludes its starting endpoint, and includes its final endpoint)
        if yPolygon1[i] > yPt and yPolygon1[i+1] <= yPt:
            # if (P is  strictly right of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) < 0:
                 wn -= 1   # a valid up intersect right of P.x

    # wn = 0 only when P is outside the polygon
    if wn == 0:
        return False
    else:
        return True

def isLeft(x1, y1, x2, y2, x3, y3):
    # isLeft(): test if a point is Left|On|Right of an infinite 2D line.
    #   From http://geomalgorithms.com/a01-_area.html
    # Input:  three points P1, P2, and P3
    # Return: >0 for P3 left of the line through P1 to P2
    # =0 for P3 on the line
    # <0 for P3 right of the line
    return ( (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) )

def rangeoverlap(a,b):
    """does the range (start1, end1) overlap with (start2, end2)"""
    return max(a) >= min(b) and min(a) <= max(b)

def doLineSegmentsIntersect(x1, y1, x2, y2):
    """ Do two line segments intersect."""

    # Check vertical lines

    # Vertical lines, but NOT same X-values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]!=x2[0]:
        return False  # No overlap

    # Vertical lines with same X values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]==x2[0]:
        # Check intersection of Y ranges
        I1 = [np.min(y1), np.max(y1)]
        I2 = [np.min(y2), np.max(y2)]

        # And we could say that Xa is included into :
        Ia = [max( np.min(y1), np.min(y2) ),
              min( np.max(y1), np.max(y2) )]

        # Now, we need to check that this interval Ia exists :
        if rangeoverlap(y1,y2)==False:
            return False  # There is no mutual abcisses
        else:
            return True   # There is overlap 

def islineliketype(line):
    """ Check if an object is line-like """
    if (sisinstance(line,'raytrace.line.Line') or sisinstance(line,'raytrace.lightray.LightRay') or
        sisinstance(line,'raytrace.line.Vector') or sisinstance(line,'raytrace.line.NormalVector') or
        sisinstance(line,'raytrace.line.Ray')):
        return True
    else:
        return False
    
def islinelike(line):
    """ Check if an object is line-like or multiple lines"""
    if islineliketype(line):
        return True
    elif (sisinstance(points,list) or sisinstance(points,tuple)) and islineliketype(line[0]):
        return True
    else:
        return False

def linelikedata(lines):
    """ Return normal vector data of line-like objects """
    if isinstance(lines,list)==False and isinstance(lines,tuple)==False:
        lines = [lines]
    data = len(lines)*[None]
    for i in range(len(lines)):
        line = lines[i]
        if sisinstance(line,'raytrace.line.Line'):
            data[i] = line.slopes.copy()
        elif sisinstance(line,'raytrace.lightray.LightRay'):
            data[i] = line.normal.data.copy()
        elif sisinstance(line,'raytrace.line.Vector') or sisinstance(line,'raytrace.line.NormalVector'):
            data[i] = line.data.copy()
        elif sisinstance(line,'raytrace.line.Ray'):
            data[i] = line.normal.data.copy()
        else:
            raise ValueError('Object is not line-like')
    data = np.atleast_2d(data)
    return data

def ispointliketype(point):
    """ Check if an object is point-like """
    if sisinstance(line,'raytrace.line.Point'):
        return True
    elif (isinstance(line,list) or isinstance(line,tuple)) and len(line)==3:
        return True
    elif isinstance(line,np.ndaray) and line.ndim==3 and line.size==3:
        return True
    else:
        return False    

def ispointlike(points):
    """ Check if an object is point-like or multiple points"""
    value = False
    if sisinstance(points,'raytrace.line.Point'):
        value = True
    elif (sisinstance(points,list) or sisinstance(points,tuple)):
        # List/tuple can have Points, numpy array or 3-element lists/tuples
        if sisinstance(points[0],'raytrace.line.Point'):
            value = True
        elif isinstance(points[0],list) or isinstance(points[0],tuple):
            value = (len(points[0])==3)
        elif isinstance(points[0],np.ndarray):
            data = np.atleast_2d(points[0])
            value = (data.shape[1]==3)
    elif isinstance(points,np.ndarray):
        data = np.atleast_2d(points)
        value = (data.shape[1]==3)
    else:
        value = False
    return value

def pointlikedata(points):
    """ Return 2D point-like data """
    if sisinstance(points,'raytrace.line.Point')>-1:
        data = np.atleast_2d(points.data)
    elif (sisinstance(points,list) or sisinstance(points,tuple)):
        # List/tuple can have Points, numpy array or 3-element lists/tuples
        if sisinstance(points[0],'raytrace.line.Point'):
            data = np.atleast_2d([p.data for p in points])
        elif isinstance(points[0],list) or isinstance(points[0],tuple):
            data = np.atleast_2d(points)
        elif isinstance(points[0],np.ndarray):
            data = np.atleast_2d(points[0])
    elif isinstance(points,np.ndarray):
        data = np.atleast_2d(points)
    else:
        raise ValueError('Object is not point(s)-like')
    return data
