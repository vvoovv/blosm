# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithmof described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220#
# 
# Implementation of the class <Segment>
#

from math import inf
from .Point import Point

class Segment():
    pSweep = None
    ID = 0
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.dx = self.p2.x - self.p1.x
        self.dy = self.p2.y - self.p1.y

        self.id = Segment.ID
        Segment.ID += 1

        if self.p2.x != self.p1.x:
            self.slope = self.dy / self.dx
            self.yShift = self.p1.y - self.slope * self.p1.x
        else: # vertical segment
            self.slope = inf
            self.yShift = -1. * inf

    def start(self):
        return self.p1

    def end(self):
        return self.p2

    def isTrivial(self):
        return self.dx == 0 and self.dy == 0

    @staticmethod
    def setpSweep(pSweep):
        Segment.pSweep = pSweep

    @staticmethod
    # see https://docs.python.org/3.0/whatsnew/3.0.html#ordering-comparisons
    # for comparison in Python >= 3.0
    def cmpVal(a, b):
        return (a > b)*1. - (a < b)*1.

    def compare(self,other):
        if self is other:
            return 0
        s = 0
        if Segment.pSweep is self.p1:
            s = Segment.orientation(other,Segment.pSweep)
        elif Segment.pSweep is other.p1:
            s = -Segment.orientation(self,Segment.pSweep)
        else:
            raise Exception('Compare error in Segment')

        if s or self.isTrivial() or other.isTrivial():
            return s
        s = Segment.orientation(other,self.p2)
        # overlapping segments will be ordered by their id-numbers
        return s if s else self.id-other.id

    @staticmethod
    def orientation(s,point):
        orient = s.dy*(point.x-s.p1.x) - s.dx*(point.y-s.p1.y)
        return Segment.cmpVal( s.dy*(s.p1.x-point.x), s.dx*(s.p1.y-point.y) )

    def intersectionOfLines(self,s):
        if self.slope == s.slope: return None
        if self.p1 == s.p1 or self.p1 == s.p2:
            return self.p1
        if self.p2 == s.p1 or self.p2 == s.p2:
            return self.p2
        if self.p1.x == self.p2.x: # is vertical
            cx = self.p1.x
        else:
            if s.p1.x == s.p2.x: # is vertical
                cx = s.p1.x
            else:
                cx = (s.yShift-self.yShift)/(self.slope-s.slope)
        if self.p1.x == self.p2.x: # is vertical
            cy = s.slope * cx + s.yShift
        else:
            cy = self.slope * cx + self.yShift
        return Point((cx,cy))

    def __gt__(self,other):
        # return self.p1 > other.p1
        return self.compare(other) > 0

    def __lt__(self,other):
        # return self.p1 < other.p1
        return self.compare(other) < 0

    def __ge__(self,other):
        # return self.p1 >= other.p1
        return self.compare(other) >= 0

    def __le__(self,other):
        # return self.p1 <= other.p1
        return self.compare(other) <= 0

    def __eq__(self,other):
        if other is None: return False
        # return self.p1 == other.p1
        return self.compare(other) == 0

    def __hash__(self):
        return self.id

    def __repr__(self):
        return '[%d: %s -> %s]'%(self.id,str(self.p1),str(self.p2))

    def plot(self,color='k'):
        import matplotlib.pyplot as plt
        self.start().plot('b')
        self.end().plot('r')
        v1 = self.start()
        v2 = self.end()
        plt.plot([v1.x,v2.x],[v1.y,v2.y],color)
        x = (v1.x+v2.x)/2
        y = (v1.y+v2.y)/2
        plt.text(x,y,str(self.id))

