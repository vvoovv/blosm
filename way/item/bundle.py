from collections import defaultdict
from mathutils import Vector

from .item import Item
from way.item.street import Street
from way.item.section import Section
from way.item.intersection import Intersection
from way.item.connectors import IntConnector


class Bundle(Item):

    ID = 1  # Must not start with zero to get unambiguous connector indices!

    def __init__(self):
        super().__init__()
        self.id = Bundle.ID
        Bundle.ID += 1

        self._pred = None
        self._succ = None

        self.streetsHead = []
        self.streetsTail = []

        self.headVerts = []
        self.tailVerts = []

    @property
    def pred(self):
        return self._pred

    @property
    def succ(self):
        return self._succ

    @pred.setter
    def pred(self,val):
        self._pred = val

    @succ.setter
    def succ(self):
        return self._succ


def makePseudoMinor(streetGenerator, intersection, arriving, leaving):
    intersection.isMinor = True

    # Find a leaving major street (by connector <conn>)
    for conn in IntConnector.iterate_from(intersection.startConnector):
        if conn.item == leaving:
            break

    intersection.leaving = leaving

    # The circular list of connectors of this intersection is
    # ordered counter-clockwise. When we start with a leaving street,
    # the first streets are to the left.
    for conn in IntConnector.iterate_from(conn.succ):
        if conn.item != arriving:
            intersection.insertLeftConnector(conn)
        else:
            break # We found the arriving street

    intersection.arriving = arriving

    # Then, the streets to the right are collected
    for conn in IntConnector.iterate_from(conn.succ):
        if conn.item != leaving:
            intersection.insertRightConnector(conn)
        else:
            break # We found leaving street again

# Utilities ===================================================================

# The order is given first by the x-coordinate of the location and then by the
# y-coordinate, if the difference in x-direction is larger than in y-direction
# and reversed, if not.
def forwardOrder(p0,p1):
    d = p1-p0
    if abs(d[0]) > abs(d[1]):   # compare x first
        return (p0[0],p0[1]) < (p1[0],p1[1])
    else:                       # compare y first
        return (p0[1],p0[0]) < (p1[1],p1[0])

def isHairpinBend(arriving, leaving):
    v0 = arriving.tail.centerline[-1] - arriving.tail.centerline[-2]
    v0 /= v0.length
    v1 = leaving.head.centerline[1] - leaving.head.centerline[0]
    v1 /= v1.length
    return v0.dot(v1) < -0.5

def locationsInGroup(streetGroup):
    streetEnds = defaultdict(list)  # Holds lists of streets at locations of their ends
    intersections = dict()                 # Holds intersection for locations
    hairpins = dict()
    for street in streetGroup:
        streetEnds[street.src].append(street)
        streetEnds[street.dst].append(street)
        intersections[street.src] = street.pred.intersection if street.pred else None
        intersections[street.dst] = street.succ.intersection if street.succ else None

    for p, streets in streetEnds.items():
        hairpins[p] = len(streets) == 2 and isHairpinBend(streets[0],streets[1])

    return streetEnds, intersections, hairpins

# see https://github.com/prochitecture/blosm/issues/104#issuecomment-2322836476
# Major intersections the street group of a bundle, with only one side street,
# are merged into a long street, similar to minor intersections.
def mergePseudoMinors(streetGenerator, streetGroup):

    def mergableIntersections(street):
        # Look for major intersections that are at the ends of streets that can be merged.
        # The conditions are:
        #   - Only 2 streets of the bundle candidate end here.
        #   - Only one extrernal street allowed (intersection order must be 3).
        #   - The streets of the bundle candidate must not build a hairpin bend.
        arrivingSrc = None
        leavingSrc = None
        arrivingDst = None
        leavingDst = None

        srcIsect = intersections[street.src]
        if srcIsect:
            if len(streetEnds[street.src])==2 and srcIsect.order==3:
                street0 = streetEnds[srcIsect.location][0]
                street1 = streetEnds[srcIsect.location][1]
                arrivingSrc, leavingSrc = (street0 if street0.dst == street.src else street1 if street1.dst == street.src else None, street)
                srcIsect = srcIsect if arrivingSrc and not isHairpinBend(arrivingSrc, leavingSrc) else None
            else:
                srcIsect = None

        dstIsect = intersections[street.dst]
        if dstIsect:
            if len(streetEnds[street.dst])==2 and dstIsect.order==3:
                street0 = streetEnds[dstIsect.location][0]
                street1 = streetEnds[dstIsect.location][1]
                arrivingDst, leavingDst = (street, street0 if street.dst == street0.src else street1 if street.dst == street1.src else None)
                dstIsect = dstIsect if leavingDst and not isHairpinBend(arrivingDst, leavingDst) else None
            else:
                dstIsect = None

        exits = {'arrivingSrc': arrivingSrc, 'leavingSrc': leavingSrc, 'arrivingDst':arrivingDst, 'leavingDst':leavingDst }
        return srcIsect, dstIsect, exits
    
    streetEnds, intersections, _ = locationsInGroup(streetGroup)

    replacedStreets = set() # Streets that have been replaced by a longer street.
    modifiedIsects = set()  # Major intersections, that have been transformed to pseudo minor ones.
    newLongStreets = []     # Merged long streets.
    for street in streetGroup:
        if street in replacedStreets:
            continue
        srcIsectIni, dstIsectIni, exitsIni = mergableIntersections(street)

        if srcIsectIni or dstIsectIni:
            # Create a new Street
            longStreet = Street(street.src,street.dst)
            longStreet.insertStreetEnd(street)
            longStreet.pred = street.pred
            replacedStreets.add(street)

            if dstIsectIni: # extend the street at its end
                dstIsectIni.street = longStreet
                arriving, leaving = exitsIni['arrivingDst'], exitsIni['leavingDst']
                makePseudoMinor(streetGenerator, dstIsectIni, arriving, leaving)
                modifiedIsects.add(dstIsectIni)
                longStreet.insertEnd(dstIsectIni)
                dstIsectCurr = dstIsectIni
                while True:
                    nextStreet = leaving
                    if nextStreet in replacedStreets:
                        break
                    longStreet.insertStreetEnd(nextStreet)
                    replacedStreets.add(nextStreet)
                    _, dstIsectCurr, currExits = mergableIntersections(nextStreet)
                    if not dstIsectCurr:
                        if nextStreet.succ is not None:
                            if isinstance(nextStreet.succ,IntConnector):
                                nextStreet.succ.item = longStreet
                        break
                    arriving, leaving = currExits['arrivingDst'], currExits['leavingDst']
                    makePseudoMinor(streetGenerator, dstIsectCurr, arriving, leaving)
                    modifiedIsects.add(dstIsectCurr)
                    dstIsectCurr.street = longStreet
                    longStreet.insertEnd(dstIsectCurr)
            else:
                if street.succ is not None:
                    if isinstance(street.succ,IntConnector):
                        street.succ.item = longStreet

            if srcIsectIni: # extend the street at its front
                srcIsectIni.street = longStreet
                arriving, leaving = exitsIni['arrivingSrc'], exitsIni['leavingSrc']
                makePseudoMinor(streetGenerator, srcIsectIni, arriving, leaving)
                modifiedIsects.add(srcIsectIni)
                longStreet.insertFront(srcIsectIni)
                srcIsectCurr = srcIsectIni
                while True:
                    if arriving in replacedStreets:
                        break
                    longStreet.insertStreetFront(arriving)
                    replacedStreets.add(arriving)
                    srcIsectCurr, _, currExits = mergableIntersections(arriving)
                    if not srcIsectCurr:
                        if arriving.pred is not None:
                            if isinstance(arriving.pred,IntConnector):
                                arriving.pred.item = longStreet
                        break
                    arriving, leaving = currExits['arrivingSrc'], currExits['leavingSrc']
                    makePseudoMinor(streetGenerator, srcIsectCurr, arriving, leaving)
                    modifiedIsects.add(srcIsectCurr)
                    srcIsectCurr.street = longStreet
                    longStreet.insertFront(srcIsectCurr)
            else:
                if street.pred is not None:
                    if isinstance(street.pred,IntConnector):
                        street.pred.item = longStreet

            newLongStreets.append(longStreet)

    # Some bookkeeping
    for street in replacedStreets:
        del streetGenerator.streets[street.id]
        streetGroup.remove(street)

    for longStreet in newLongStreets:
        streetGenerator.streets[longStreet.id] = longStreet
        streetGroup.append(longStreet)

    for isect in modifiedIsects:
        del streetGenerator.majorIntersections[isect.id]
        streetGenerator.minorIntersections[isect.id] = isect

    return streetGroup


def removeIntermediateSections(streetGroup):
    streetEnds, intersections, hairpins = locationsInGroup(streetGroup)

    # Check if intermediate intersections exist
    hasIntermediates = any(len(end)>1 for end in streetEnds.values())

    hadIntermediates = hasIntermediates
    while hasIntermediates:
        hasIntermediates = False
        for p,streets in streetEnds.items():
            if len(streets)==2 and not hairpins[p]:
                smallestStreet = min(streets, key = lambda x: x.length())
                if smallestStreet in streetGroup:
                    streetGroup.remove(smallestStreet)
                print(smallestStreet.id, 'removed')
                streetEnds, intersections, hairpins = locationsInGroup(streetGroup)
                hasIntermediates = any(len(end)>1 for end in streetEnds.values())
                break

    return streetGroup

def orderHeadTail(streetGroup):
    streetEnds, intersections, hairpins = locationsInGroup(streetGroup)

    # Try to find starts and ends of the streets relative to the bundle. This 
    # 'forwardOrder' determines also the direction of the bundle. See this
    # function above.
    head = []
    tail = []
    for street in streetGroup:
        fwd = forwardOrder(street.src,street.dst)
        h, t = (street.src, street.dst) if fwd else (street.dst, street.src)
        if len(streetEnds[h])==1 or hairpins[h]:  
                head.append({'street':street, 'firstVert':h, 'lastVert':t})
        if len(streetEnds[t])==1 or hairpins[t]:  
                tail.append({'street':street, 'firstVert':t, 'lastVert':h})

    # The streets in head and tail have to be ordered from left to right.
    # The vector of the first segment of an arbitrary street is turned 
    # perpendicularly to the right and all points at this end of the bundle
    # are projected onto this vector, delivering some kind of a line parameter t.
    # The lines are then sorted by increasing values of this parameter, which
    # orders them from left to right, seen in the direction of the bundle.

    # Process head (start) of the future bundle
    arbStreet = head[0]['street']   # arbitrary street at the bundle's head
    # forwardVector points from the head into the bundle.
    fwd = forwardOrder(arbStreet.src,arbStreet.dst)
    srcVec, dstVec = arbStreet.endVectors()
    forwardVector = srcVec if fwd else dstVec
    
    # the origin of this vector
    p0 = arbStreet.src if fwd else arbStreet.dst

    # perp is perpendicular to the forwardVector, turned to the right
    perp = Vector((forwardVector[1],-forwardVector[0])) 

    # sort streets along perp from left to right
    sortedIndices = sorted( (i for i in range(len(head))),  key=lambda i: ( head[i]['firstVert'] - p0).dot(perp) )

    for k, indx in enumerate(sortedIndices):
        head[indx]['i'] = k

    # Process tail (end) of the bundle
    # take an aribtrary street
    arbStreet = tail[0]['street']   # arbitrary street at the bundle's tail

    # forwardVector point from tail out of the bundle
    fwd = forwardOrder(arbStreet.src,arbStreet.dst)
    srcVec, dstVec = arbStreet.endVectors()
    forwardVector = -dstVec if fwd else -srcVec
    
    # the origin of this vector
    p0 = arbStreet.dst if fwd else arbStreet.src

    # perp is perpendicular to forwardVector, turned to the right
    perp = Vector((forwardVector[1],-forwardVector[0])) 

    # sort streats along perp from left to right
    sortedIndices = sorted( (i for i in range(len(tail))),  key=lambda i: ( tail[i]['lastVert'] - p0).dot(perp) )
                        
    for k, indx in enumerate(sortedIndices):
        tail[indx]['i'] = k

    return head, tail

def findInnerStreets(streetGroup,leftHandTraffic):
    def innerNodes(street):
        nodeIDs = set()
        for item in street.iterItems():
            if isinstance(item,Intersection):
                nodeIDs.add(item.id)
        return nodeIDs

    def lastIsectSeenFrom(node, street):
        if node.location == street.src:
            return street.succ.intersection
        else:
            return street.pred.intersection
        
    # Find all intersections that lead to inner streets
    innerIsects = dict()
    for street in streetGroup:
            for item in street.iterItems():
                if isinstance(item,Intersection):
                        if not item.isMinor:
                            print('!!!!!!')
                        # Inner streets leave to the left if lefthand traffic, else to the right
                        iterator = Intersection.iterate_from(item.leftHead) if leftHandTraffic else \
                                   Intersection.iterate_from(item.rightHead)
                        for intConn in iterator:
                            innerIsects[intConn.intersection] = intConn.item

    from debug import plt, plotEnd
    for intersection, street in innerIsects.items():
        p = intersection.location
        plt.plot(p[0],p[1],'co',markersize=8)
        # for item in street.iterItems():
        #     if isinstance(item, Section):
        #         item.polyline.plot('c',2,'solid',False,999)

    innerStreets = set()
    bundleIsects = set()
    for intersection, innerStreet in innerIsects.items():
        innerStreets.add(innerStreet)
        # Maybe this inner street has inner (minr) intersections
        # bundleIsects.union( innerNodes(innerStreet) )
        lastIsect = lastIsectSeenFrom(intConn.intersection,innerStreet)
        if lastIsect not in innerIsects:
            bundleIsects.add(lastIsect)

    for intersection in bundleIsects:
        p = intersection.location
        plt.plot(p[0],p[1],'mo',markersize=8,zorder=800)



    additionalStreets = set()
    for item in bundleIsects:
        if item.isMinor:
            for intConn in Intersection.iterate_from(item.leftHead):
                if intConn.item not in innerStreets:
                    innerStreets.add(intConn.item)
                    additionalStreets.add(intConn.item)
            for intConn in Intersection.iterate_from(item.rightHead):
                if intConn.item not in innerStreets:
                    innerStreets.add(intConn.item)
                    additionalStreets.add(intConn.item)
            if item.street not in innerStreets:
                innerStreets.add(item.street)
                additionalStreets.add(item.arriving)
        else:
            for intConn in item:
                if intConn.item not in innerStreets:
                    innerStreets.add(intConn.item)
                    additionalStreets.add(intConn.item)

    return innerStreets
