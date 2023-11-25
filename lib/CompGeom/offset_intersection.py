from itertools import product
from mathutils import Vector

def offsetIntersection(e0,e1,w0,w1,lim=1.e-3):
    # Computes the intersection of a right edge <e0>, expanded to the
    # left by <w0>, and a left edge <e1>, expanded to the right by <w1>.
    # Inputs:
    # e0: rigth edge given by a tuple (v0,v1) of vectors (mathutils.Vector).
    # e1: left edge given by a tuple (v0,v1) of vectors (mathutils.Vector).
    # w0: buffer width of <e0> to the left.
    # w1: buffer width of <e1> to the right.
    # lim: minimum angle to declare as parallel: lim = sin(angle)
    # Returns: 
    # isect: Intersection point as mathutils.Vector.
    # valid: - String 'valid', if isect is valid.
    #        - String 'out', if intersection is out of normals at the upper
    #          end of the edges.
    #        - String 'parallel', if angle between the edges is less than
    #          a value given by <lim>: angle < asin(lim)
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # edge vectors and their unit vectors
    ev0,ev1 = e0[1]-e0[0], e1[1]-e1[0]
    e0u, e1u = -ev0/ev0.length, ev1/ev1.length
    # cross product is sine of angle between edge vectors
    if abs(e1u.cross(e0u)) < lim:
        return Vector((0,0)), 'parallel', False
    # unit normals of the edges
    n0,n1 = Vector((e0u[1],-e0u[0])), Vector((e1u[1],-e1u[0]))
    # width offset of <e1> when edges start at different positions
    dw = (e1[0]-e0[0]).dot(n1)

    # The construction of the buffer intersection is similar to the motion
    # vector in a wighted straight skeleton algorithm. The following computation
    # is adapted from the master thesis of Gerhild Grinschgl,
    # https://diglib.tugraz.at/weighted-straight-skeleton-grundlagen-und-implementierung-2016
    # 2016, page 39ff.

    # expanded normals
    n0w,n1w = n0*w0, n1*(w1+dw)
    u0 = ev1.cross(ev0)
    u = ev1.cross(n1w-n0w) / u0
    isect = e0[0] + n0w + ev0*u
    # Test if <isect> is out of normals at the upper end of the edges.
    valid = ccw(e0[1],e0[1]+n0,isect) and not ccw(e1[1],e1[1]+n1,isect)
    isPos = u >= 0.
    kind = 'valid' if valid else 'out'
    return isect, kind, isPos

def offsetPolylineIntersection(line0,line1,w0,w1,exitOnParallel=False,lim=1.e-3):
    # Find the intersection of a right polyline <line0>, offset by <w0> to the left, and
    # a left polyline <line1>, offset to the right by <w1>.
    #           
    #         line1 /  w1  /
    #              /      /
    #             /      /
    #            /      /
    #           /      /  X is <isect>
    #          /      X----------------------
    #         /
    #        /                          w0
    #       /
    #      o--------------------------------- line0
    #
    # Returns:
    # isect: Intersection point as mathutils.Vector.
    # kind:  - String 'valid', if isect is valid.
    #        - String 'out', if intersection is out of normals at the upper
    #          end of the lines.
    #        - String 'parallel', if angle between the lines is less than
    #          a value given by <lim>: angle < asin(lim)
    segs0, segs1 = line0.segments(),line1.segments()
    combinations = list(product(range(len(segs0)),range(len(segs1))))
    combinations = sorted(combinations,key=lambda x: sum(x))
    breakCond, contCond = (['valid','parallel'],['out']) if exitOnParallel else (['valid'],['out','parallel'])

    isFirst = True 
    result = (Vector((0,0)), '')
    for i0,i1 in combinations:
        isect, kind, isPos = offsetIntersection(segs0[i0],segs1[i1],w0,w1,lim) 
        if kind in breakCond:
            if isFirst: # Store the first intersection parameters
                result = (isect, kind)
                isFirst = False
                if kind == 'parallel':
                    break
            if isPos:   # Overwrite result, if there is a better intersection than the first
                result = (isect, kind)
                break
            continue
        elif kind in contCond:
            result = result if result[1] == 'valid' else (Vector((0,0)), kind)
            continue

    return result

