# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithmof described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220#
# 
# Implementation of the class <Point>
#

class Point():
    ID = 0
    EPS = 0.00001
    EPS2 = EPS*EPS
    def __init__(self,p):
        self.x = p[0]
        self.y = p[1]
        self.id = Point.ID
        Point.ID += 1

    def compare(self,other):
        if self is other: return 0
        dx = self.x - other.x
        if dx >  Point.EPS2: return  1
        if dx < -Point.EPS2: return -1
        dy = self.y - other.y
        if dy >  Point.EPS2: return  1
        if dy < -Point.EPS2: return -1
        return 0

    def __gt__(self,other):
        return self.compare(other) > 0

    def __lt__(self,other):
        return self.compare(other) < 0

    def __ge__(self,other):
        return self.compare(other) >= 0

    def __le__(self,other):
        return self.compare(other) <= 0

    def __eq__(self,other):
        if other is None: return False
        return self.compare(other) == 0

    def __iter__(self):
        # used to create tuples
        return iter([self.x,self.y])

    def __repr__(self):
        return '(%d: %6.2f,%6.2f)'%(self.id,self.x,self.y)

    def plot(self,color='k',size=3):
        import matplotlib.pyplot as plt
        plt.plot(self.x,self.y,color+'o',markersize=size)
        plt.text(self.x,self.y,str(self.id))
