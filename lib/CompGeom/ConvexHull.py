# Implementation of a Quick Hull algorithm, to find the convex hull
# of a cloud of points, represented as mathutils.Vector.
# see: https://en.wikipedia.org/wiki/Quickhull
#
# Usage:
#       hullCreator = ConvexHull()
#       hull = hullCreator.convexHull(points)
#

from mathutils import Vector
from math import atan2

class ConvexHull():
    def __init__(self):
        self.points = None
        self.hull = set()

    def convexHull(self, points):
        if points is None or len(points) < 3:
            print('Invalid point set for convex hull')
            return set(points)
        if len(points) == 3:
            return set(points)
        _points = [p.freeze() for p in points]          
        self.points = list(dict.fromkeys(_points))   # Remove duplicates
        self.hull = set()

        # Find the point with minimum and maximum x-coordinate
        iMin = min(range(len(self.points)), key=lambda i: self.points[i][0])
        iMax = max(range(len(self.points)), key=lambda i: self.points[i][0])

        # Recursively find convex hull points on both sides of the line
		# joining points[iMin] and points[iMax]
        self.quickHull(points, points[iMin], points[iMax], 1)
        self.quickHull(points, points[iMin], points[iMax], -1)
        return ConvexHull.sortCounterClockwise(self.hull)
	
    @staticmethod
    def ccw(p1, p2, p):
        val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        return 1 if val>0. else -1 if val < 0. else 0
	
    @staticmethod
    def dist(p1, p2, p):
        return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))
	
    @staticmethod
    def sortCounterClockwise(points):
        # Return sorted vertices in counter-clockwise order by sorting angles
        # seen from the center of gravity.
        def _pseudoangle(d):
            p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
            return 3 + p if d[1] < 0 else 1 - p 
        
        if points and len(points) > 1:
            centerOfGravity = sum( (p for p in points),Vector((0,0))) / len(points)
            return sorted(points,key=lambda x: _pseudoangle(x-centerOfGravity))
        else:
            return points

    def quickHull(self, points, p1, p2, side):
	    # Find the point with maximum distance from p1-p2 and also on the specified side of p1-p2.
        ind = -1
        max_dist = 0
        for i,p in enumerate(points):
            temp = ConvexHull.dist(p1, p2, p)        
            if (ConvexHull.ccw(p1, p2, p) == side) and (temp > max_dist):
                ind = i
                max_dist = temp
        # If no point is found, add the end points of p1-p2 to the convex hull.
        if ind == -1:
            self.hull.add(p1)
            self.hull.add(p2)
            return

        # Recur for the two parts divided by a[ind]
        pMax = points[ind]
        self.quickHull(points, pMax, p1, -ConvexHull.ccw(pMax, p1, p2))
        self.quickHull(points, pMax, p2, -ConvexHull.ccw(pMax, p2, p1))


