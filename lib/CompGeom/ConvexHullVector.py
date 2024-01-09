# Adapted from ConvexHull2D of Zifan Wang: https://github.com/zifanw/ConvexHull2D
# Implementation of a Quickhull Algorithm for 2D points.

from mathutils import Vector
from math import atan2

def divide2Subsets(start, end, points):
    # Divide the bag of points relative to the line start-end into two bags 
    # by comparing the cross product.
    # start: The start point of the boundary line
    # end: The end point of the boundary line
    # points: The bag of points to be divided
    if not points:
        return None, None
    S1, S2 = [], []
    for point in points:
        dis =  computeDistance(start, end, point)
        if dis > 0:
            S1.append(point)
        else:
            S2.append(point)
    S1 = S1 if len(S1) else None
    S2 = S2 if len(S2) else None
    return S1, S2

def computeDistance(start, end, point, eps=1e-8):
    # Return the cross product from point to segment defined by (start, end)
    return (end-start).cross(point-start)/((end-start).length + eps) # prevent division by 0

def sortClockwise(points):
    # Return sorted vertices in clockwise order using the angle 
    # between the x-axis and the vector pointing from the center of mass
    # to the point x
    center = sum((p for p in points), Vector((0,0)) ) / len(points)
    vectors = [v-center for v in points]
    angles = [atan2(v[1],v[0]) for v in vectors]
    #argsort: http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    index = sorted(range(len(angles)), key=angles.__getitem__)
    points = [points[i] for i in index]
    return points

class ConvexHullVector:
    # This function returns an array containing the locations of points which 
    # form a convex hull for a 2D cloud of points (list of mathutils.Vector). 
    # Sample Usage:
    #
    #     #Initilization
    #     ch_creator = ConvexHullVector()
    #
    #     #find the convex hull
    #     points = np.random.random((100,2)) 
    #     hull = ch_creator(points)
    #
    #     #plot
    #     plt.plot(hull[:,0], hull[:,1])
    #     plt.show()
    def __init__(self):
        self.points = None
        self.convex_hull = []
        self.valid = False
        self.errorText = ''

    def __call__(self, points):
        return self.forward(points)

    def _reset(self):
        self.points = None
        self.convex_hull = []
        self.valid = False
        self.errorText = ''

    @property
    def isValid(self):
        return self.valid

    def forward(self, points):
        if points is None or len(points) < 3:
            self.errorText = 'No valid convex hull is found! Please provide more than 3 unique points'
            self.valid = False
            return None
        self._reset()
        _points = [p.freeze() for p in points]          
        self.points = list(dict.fromkeys(_points))   # Remove duplicates
        hull = self._quickhull()
        self.valid = hull is not None and len(hull)>1
        return hull
    
    def isInside(self, points):
        if not self.convex_hull:
            self.errorText = 'Please build a convex hull first.'
            self.valid = False
            return None
        return [self._isInside(point) for point in points ]   

    def _quickhull(self):        
        self.points.sort(key=lambda p: (p[0],p[1]))   # Sort the data by x-axis, then by y-axis
        left_most, right_most = self.points[0], self.points[-1]     # Find the left-most point and right-most point
        self.points = self.points[1:-1]                             # Get the remaining points
        self.convex_hull.append(left_most)     # Add the left-most point into the output
        self.convex_hull.append(right_most)    # Add the right-most point into the output

        self.right_points, self.left_points = divide2Subsets( left_most, right_most, self.points)

        self._findhull(self.right_points, left_most, right_most)
        self._findhull(self.left_points, right_most, left_most)

        self.convex_hull = sortClockwise(self.convex_hull)
        if len(self.convex_hull) < 3:
            self.valid = False
            self.errorText = 'Not enough points are found for convex hull. Please check your input and other information'
            return None
        else:                   
            return self.convex_hull

    def _findhull(self, points, P, Q):
        if points is None:
            return None
        distance = 0.0
        C, index = None, None
        for point in points:
            current_dis = abs(computeDistance(P, Q, point))
            if current_dis > distance: # Find a point whose distance from PQ is the maximum among all the points
                C = point
                distance = current_dis
        if C is not None:
            self.convex_hull.append(C)
            points = points.remove(C)   # Delete C from original points
        else:
            self.valid = False
            self.errorText = 'The input points are located on the same line. No convex hull is found!'
            return

    def _isInside(self, point):
        for i in range(len(self.convex_hull)-1):
            start, end = self.convex_hull[i], self.convex_hull[i+1]
            if computeDistance(start, end, point) < 0:
                return False
        return True
