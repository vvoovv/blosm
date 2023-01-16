# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithm described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220
# 
# Implementation of the main class <SweepIntersector> that implements the
# sweep line algorithm for line-segment intersection.
#

from collections import defaultdict

from .SortSeq import SortSeq
from .PriorityQueue import PriorityQueue
from .Segment import Segment
from .Point import Point

class SweepIntersector():
    def __init__(self):
        self.X_structure = SortSeq()
        self.Y_structure = SortSeq()
        self.lastNode = dict()
        self.original = dict()
        self.assoc = dict()
        self.interDic = dict()
        self.segQueue = PriorityQueue()
        self.pSweep = None
        self.N = 0
        self.isectDict = defaultdict(list)
        self.intersectingSegments = defaultdict(list)

    def findIntersections(self,origSegList):
        """
        Main method. Computes all intersections between a list <origSegList> of
        segments.
        <origSegList>: List of tuples (vs,ve) for segments, where vs is the start
                       point and ve the end point. The points v1 and v2 are given
                       as tuples (x,y) where x and y are their coordinates in the
                       plane. 
        Returns:       A dictionary <seg:isects> for all segments that had inter-
                       sections. <seg>, the key of the dictionary, is a tuple 
                       (vs,ve) identical to the one in the input list and <isects>
                       is the list of the intersections points. These points
                       are given tuples (x,y) where again x and y are their
                       coordinates in the plane. This list includes the start and
                       end points vs and ve and is ordered from vs to ve.

        Usage example:

            from SweepIntersectorLib import SweepIntersector

            origSegList = []
            origSegList.append( ((1.0,1.0),(5.0,6.0)) )
            origSegList.append( ((1.0,4.0),(4.0,0.0)) )
            origSegList.append( ((1.5,5.0),(3.0,1.0)) )
            ...

            isector = SweepIntersector()
            isectDic = isector.findIntersections(origSegList)
            for seg,isects in isectDic.items():
                ...

        """
        self.initializeStructures(origSegList)

        # Main loop
        while not self.X_structure.empty():
            # move <pSweep> to next event point.
            event = self.X_structure.min()
            Segment.pSweep = self.pSweep = self.X_structure.key(event)
            v = self.pSweep

            # self.G.append(self.pSweep);
            # print('GRAPH')
            # for elem in self.G:
            #     print(elem)
            # print(self.Y_structure)
            # p = self.Y_structure.head
            # while p.succ[0].key:
            #     node = p.succ[0]
            #     if node.data:
            #         print('*')
            #         print('k--> ',node.key)
            #         print('d--> ',node.data,type(node.data))
            #         print('dk---> ',node.data.key)
            #         print('dd---> ',node.data.data,type(node.data.data))
            #     p = p.succ[0]
            # for key,val in self.interDic.items():
            #     print('dic',key, val.key)
            # self.plotAll()

            # If there is an item <sit> associated with <event>,
            # key(sit) is either an ending or passing segment.
            # We use <sit> as an entry point to compute the bundle of
            # segments ending at or passing through <pSweep>.
            # In particular, we compute the first <sitFirst> and last
            # <sitLast> item of this bundle and the successor <sitSucc>
            # and predecessor <sitPred> items.
            sit = SortSeq.inf(event)

            if sit is None:
                # Here we do not know any segments ending or passing through 
                # <pSweep>. However, <pSweep> could come to lie on a segment 
                # inserted before. To check this we look up the zero length 
                # segment (pSweep,pSweep).
                sit = self.Y_structure.lookup(Segment(self.pSweep,self.pSweep))

            sitSucc  = None
            sitPred  = None
            sitFirst = None
            sitLast  = None

            # A value of None for <sitSucc> and <sitPred> after the 
            # following computation indicates that there are no segments 
            # ending at or passing through <pSweep>

            if sit: # key(sit) is an ending or passing segment
                # first walk up until <sitSucc>
                while self.Y_structure.inf(sit) == event:
                    sit = self.Y_structure.succ(sit)
                sitSucc = self.Y_structure.succ(sit)
                xit = self.Y_structure.inf(sit)
                if xit: 
                    s1 = self.Y_structure.key(sit)
                    s2 = self.Y_structure.key(sitSucc)
                    self.interDic[(s1.id,s2.id)] = xit

                # Walk down until <sitPred>, construct edges for all segments
                # in the bundle, assign information <None> to continuing segments,
                # and delete ending segments from the Y_structure
                while True:
                    s = self.Y_structure.key(sit)
                    sr = self.assoc[s]

                    self.lastNode[s] = v
                    if self.pSweep is s.p2: #  ending segment
                        it = self.Y_structure.pred(sit)
                        self.Y_structure.delItem(sit)
                        sit = it;
                    else:   # passing segment
                        self.Y_structure.changeInf(sit,None)
                        sit = self.Y_structure.pred(sit)
                        if (sr is not s) and (sr.p2 is self.pSweep):
                            self.assoc[s] = s

                    if self.Y_structure.inf(sit) != event:
                        break # end of while True:

                sitPred  = sit
                sitFirst = self.Y_structure.succ(sitPred)

                # reverse the bundle of continuing segments (if existing)
                if sitFirst != sitSucc:
                    sitLast = self.Y_structure.pred(sitSucc)
                    self.Y_structure.reverseItems(sitFirst,sitLast)

            # Insert all segments starting at <pSweep>
            while self.pSweep is self.nextSeg.start():  # identity
                # insert <nextSeg> into the Y-structure and associate the
                # corresponding item with the right endpoint of <nextSeg> in
                # the X-structure (already present)
                sit = self.Y_structure.locate(self.nextSeg)
                seg0 = self.Y_structure.key(sit)

                if self.nextSeg != seg0:
                    # <next_seg> is not collinear with <seg0>, insert it
                    sit = self.Y_structure.insertAt(sit, self.nextSeg, None)
                    self.X_structure.insert(self.nextSeg.end(),sit)
                    self.lastNode[self.nextSeg] = v

                    if sitSucc is None:
                        # There are only starting segments, compute <sitSucc>
                        # and <sitPred> using the first inserted segment
                        sitSucc = self.Y_structure.succ(sit)
                        sitPred = self.Y_structure.pred(sit)
                        sitFirst = sitSucc
                else:
                    # <nextSeg> and <seg0> are collinear; if <next_seg> is
                    # longer insert (seg0.end(),next_seg.end()) into <segQueue>
                    print('XXXXXXXXXXXX Collinear segments in intersector')
                    p = seg0.end()
                    q = self.nextSeg.end()
                    self.assoc[seg0] = self.nextSeg
                    if p < q:
                        newSeg = Segment(p,q) 
                        self.assoc[newSeg] = newSeg
                        self.original[newSeg] = self.original[self.nextSeg]
                        self.segQueue.insert(p,newSeg)

                # delete minimum and assign new minimum to <nextSeg>
                self.segQueue.delMin()
                self.nextSeg = self.segQueue.inf(self.segQueue.min())

            # if <sitPred> still has the value <None>, there were no ending, 
            # passing or starting segments, i.e., <pSweep> is an isolated 
            # point. In this case we are done, otherwise we delete the event 
            # associated with <sitPred> from the X-structure and compute 
            # possible intersections between new neighbors.
            if sitPred is not None:
                # <sitPred> is no longer adjacent to its former successor we 
                # change its intersection event to None.
                xit = self.Y_structure.inf(sitPred) 

                if xit is not None: 
                    s1 = self.Y_structure.key(sitPred)
                    s2 = self.Y_structure.key(sitFirst)
                    self.interDic[(s1.id,s2.id)] = xit
                    self.Y_structure.changeInf(sitPred, None)

                # compute possible intersections between <sitPred> and its
                # successor and <sit_succ> and its predecessor
                self.computeIntersection(sitPred)
                sit = self.Y_structure.pred(sitSucc)
                if sit != sitPred:
                    self.computeIntersection(sit)
            self.X_structure.delItem(event)

        self.collectAndSortResult()
        return self.intersectingSegments
 
    def initializeStructures(self,origSegList):
        """
        Initializes the class using the provided list of segments <origSegList>.
        A vertex <v> is represented as a tuple (x,y).
        A segment <s> is represented by a tuple of vertices (vs,ve), where <vs> is the 
        starting point and <ve> the end point. <origSegList> is a list of segments <s>.
        """
        infinity = 1
        for segIndex, seg in enumerate(origSegList):
            v1, v2 = seg

            # Compute an upper bound |Infinity| for the input coordinates
            while abs(v1[0]) >= infinity or abs(v1[1]) >= infinity or \
                  abs(v2[0]) >= infinity or abs(v2[1]) >= infinity:
                infinity *= 2;

            it1 = self.X_structure.insert(Point(seg[0]),None)
            it2 = self.X_structure.insert(Point(seg[1]),None)
            if it1 == it2: continue  # Ignore zero-length segments

            # Insert operations into the X-structure leave previously
            # inserted points unchanged to achieve that any pair of
            # endpoints <p1> and <p2> with p1 == p2 are identical.
            p1 = SortSeq.key(it1)
            p2 = SortSeq.key(it2)
            s = Segment(p1,p2) if p1 < p2 else Segment(p2,p1)

            # use maps to associate with every segment its original
            self.original[s] = (seg,segIndex)
            self.assoc[s] = s

            # for every created segment (p1,p2) insert the pair (p1,(p1,p2)) 
            # into priority queue <segQueue>
            self.segQueue.insert(s.start(),s)

        # insert a lower and an upper sentinel segment to avoid special
        # cases when traversing the Y-structure
        lowerSentinel = Segment( Point((-infinity,-infinity)), Point((infinity,-infinity)))
        upperSentinel = Segment( Point((-infinity, infinity)), Point((infinity, infinity)))

        # the sweep begins at the lower left corner
        Segment.pSweep = self.pSweep = lowerSentinel.start()
        self.Y_structure.insert(upperSentinel,None);
        self.Y_structure.insert(lowerSentinel,None);

        # insert a stopper into <segQueue> and initialize |next_seg| with
        # the first segment in the queue.
        pStop = Point((infinity,infinity))
        sStop = Segment(pStop,pStop)
        self.segQueue.insert(pStop,sStop)
        self.nextSeg = self.segQueue.inf(self.segQueue.min())
        self.N = sStop.id

    def computeIntersection(self,sit0):
        # Given an item <sit0> in the Y-structure compute the point of 
        # intersection with its successor and (if existing) insert it into 
        # the event queue and do all necessary updates.
        sit1 = self.Y_structure.succ(sit0)
        s0   = self.Y_structure.key(sit0)
        s1   = self.Y_structure.key(sit1)

        # <s1> is the successor of <s0> in the Y-structure, hence,
        # <s0> and <s1> intersect right or above of the sweep line
        # if (s0.start(),s0.end(),s1.end() is not a left turn and 
        # (s1.start(),s1.end(),s0.end() is not a right turn.
        # In this case we intersect the underlying lines
        if Segment.orientation(s0,s1.end()) <= 0 and Segment.orientation(s1,s0.end()) >= 0:
            it = self.interDic.get((s0.id,s1.id),None)
            if it is not None:
                self.Y_structure.changeInf(sit0,it)
                del self.interDic[(s0.id,s1.id)]
            else:
                q = s0.intersectionOfLines(s1)
                if q:
                    self.Y_structure.changeInf(sit0, self.X_structure.insert(q,sit0))

                    # insert intersection point into result dictionary
                    if s0.p1 != q and s0.p2 != q:
                        self.isectDict[self.original[s0]].append((q.x,q.y))
                    if s1.p1 != q and s1.p2 != q:
                        self.isectDict[self.original[s1]].append((q.x,q.y))

    @staticmethod
    def inorderExtend(segment, v1, v2, points):
        # Extend a segment <segment> by <points> that are on
        # between the points v1, v2
        k, r = None, False
        if v1[0] < v2[0]:   k, r = lambda i: i[0], True
        elif v1[0] > v2[0]: k, r = lambda i: i[0], False
        elif v1[1] < v2[1]: k, r = lambda i: i[1], True
        else:               k, r = lambda i: i[1], False
        l = [ p for p in sorted(points, key=k, reverse=r) ]
        i = next((i for i, p in enumerate(segment) if p == v2), -1)
        assert(i>=0)
        for e in l:
            # a vertex can appear only once in a segment
            if not e in segment:
                segment.insert(i, e)
        return segment

    def collectAndSortResult(self):
        for seg,isects in self.isectDict.items():
            v1,v2 = seg[0]
            segment = self.inorderExtend([v1,v2],v1,v2,isects)
            self.intersectingSegments[seg[0]] = segment
            
    def plotY(self):
        import matplotlib.pyplot as plt
        for node in self.Y_structure._level():
            node.key.plot()
            if node.data:
                node.data.plot('m')
        plt.gca().axis('equal')
        plt.show()

    def plotResult(self):
        import matplotlib.pyplot as plt
        plt.close()
        for key,value in self.original.items():
            v1,v2 = key.p1,key.p2
            plt.plot([v1.x,v2.x],[v1.y,v2.y],'k:')
            plt.plot(v1.x,v1.y,'k.')
            plt.plot(v2.x,v2.y,'k.')
            # plt.text(v1.x,v1.y,str(v1.id))
        # for isect in self.isects:
        #     plt.plot(isect.x,isect.y,'ro',markersize=3,zorder=10)
        # plt.gca().axis('equal')
        # plt.show()

    def plotAll(self):
        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        for isect in self.isects:
            plt.plot(isect.x,isect.y,'rx',markersize=12)
        count = 0
        for key,value in self.original.items():
            v1,v2 = key.p1,key.p2
            plt.plot([v1.x,v2.x],[v1.y,v2.y],'k')
            plt.text(v1.x,v1.y,str(v1.id))
            count += 1
        plt.gca().axis('equal')
        plt.gca().set_title('Original Segs')
        plt.subplot(2,2,2)
        for s in self.original.keys():
            v1,v2 = s.p1, s.p2
            plt.plot([v1.x,v2.x],[v1.y,v2.y],'k')
            x = (v1.x+v2.x)/2
            y = (v1.y+v2.y)/2
            plt.text(x,y,str(s.id))
        plt.gca().axis('equal')
        plt.gca().set_title('Segments')
        plt.subplot(2,2,3)
        for node in self.X_structure._level():
            node.key.plot()
            if node.data:
                node.data.key.plot()
        plt.plot(self.pSweep.x,self.pSweep.y,'co',markersize=8)
        plt.gca().axis('equal')
        plt.gca().set_title('X-Structure')
        plt.subplot(2,2,4)
        for node in self.Y_structure._level():
            node.key.plot()
            if node.data:
                if isinstance(node.data.key,Segment):
                    node.data.key.plot('m')
                else:
                    node.data.key.plot('m',10)
        plt.plot(self.pSweep.x,self.pSweep.y,'co',markersize=8)
        for isect in self.isects:
            plt.plot(isect.x,isect.y,'rx',markersize=3)
        plt.gca().axis('equal')
        plt.gca().set_title('Y-Structure')
        plt.show()
