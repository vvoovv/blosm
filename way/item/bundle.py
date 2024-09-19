from collections import defaultdict
from itertools import tee, islice, cycle
from mathutils import Vector

from .item import Item
from way.item.street import Street
from way.item.section import Section
from way.item.intersection import Intersection
from way.item.connectors import IntConnector

# helper functions -----------------------------------------------
def cyclePairs(lst):
    prevs, nexts = tee(lst)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    return zip(prevs,nexts)

def _pseudoangle(d):
    p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
    return 3 + p if d[1] < 0 else 1 - p 
# ----------------------------------------------------------------


class Bundle(Item):

    ID = 1  # Must not start with zero to get unambiguous connector indices!

    def __init__(self):
        super().__init__()
        self.id = Bundle.ID
        Bundle.ID += 1

        self._pred = None
        self._succ = None

        # References to bundle streets at arbitrary start (head) and end (tail) of the bundle.
        # These are ordered from left to right, relative to this direction of the bundle.
        self.streetsHead = []
        self.streetsTail = []

        # References to locations of street intersections at the head and the tail of the bundle.
        # The same order as for the streets above is used.
        self.headLocs = []
        self.tailLocs = []

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
    def succ(self,val):
        self._succ = val


def makePseudoMinor(streetGenerator, intersection, arriving, leaving):
    intersection.isMinor = True
    if intersection.location in streetGenerator.minorIntersections:
        del streetGenerator.majorIntersections[intersection.location]
    streetGenerator.minorIntersections[intersection.location] = intersection

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
        #   - Only one external street allowed (intersection order must be 3).
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
        if isect.id in streetGenerator.majorIntersections:
            del streetGenerator.majorIntersections[isect.location]
        streetGenerator.minorIntersections[isect.location] = isect

def removeIntermediateSections(gIndex, streetGroup, groupIntersections):
    # <groupIntersections> holds intersections between streets 
    # of different groups. The dictionary key is the position
    # of the intersection and its  lists contain the indices
    # of the groups that intersect there.
    

    # Streets in <streetGroup>, that have intersections with other bundles 
    # than the current bundle with the index <gIndex>, but with from a 
    # common bundle on either end of the street, are normally streets
    # from large intersections, that have been included by createParallelStreets().
    #
    # They are removed from <streetGroup>, and two new groups <srcGroup> and 
    # <dstGroup> are created.
    intermediateStreets = []
    streetEnds, _, hairpins = locationsInGroup(streetGroup)
    for street in streetGroup:
        if street.src in groupIntersections and street.dst in groupIntersections:
            srcSet = {indx for indx in groupIntersections[street.src] if indx != gIndex}
            dstSet = {indx for indx in groupIntersections[street.dst] if indx != gIndex}
            commonSet = srcSet.intersection(dstSet)
            if commonSet:
                intermediateStreets.append(street)

    srcGroup = []
    dstGroup = []
    wasSplitted = False
    # Do not remove the complete group, only part of it
    if intermediateStreets and len(intermediateStreets) < len(streetGroup):
        wasSplitted = True
        for street in intermediateStreets:
            fwd = forwardOrder(street.src,street.dst)
            h, t = (street.src, street.dst) if fwd else (street.dst, street.src)
            srcGroup.extend([s for s in streetGroup if s!=street and h in [s.src, s.dst]])
            dstGroup.extend([s for s in streetGroup if s!=street and t in [s.src, s.dst]])
            print(street.id, 'removed')
    
    for group in ([srcGroup, dstGroup] if wasSplitted else [streetGroup]):
        # Check if intermediate intersections exist
        streetEnds, _, hairpins = locationsInGroup(group)
        hasIntermediates = any(len(end)>1 for end in streetEnds.values())

        while hasIntermediates:
            hasIntermediates = False
            for p,streets in streetEnds.items():
                if len(streets)==2 and not hairpins[p]:
                    smallestStreet = min(streets, key = lambda x: x.length())
                    if smallestStreet in group:
                        group.remove(smallestStreet)
                    print(smallestStreet.id, 'removed')
                    streetEnds, _, hairpins = locationsInGroup(group)
                    hasIntermediates = any(len(end)>1 for end in streetEnds.values())
                    break

    return wasSplitted, srcGroup, dstGroup

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

    reverse = False
    head = [h for (_, h) in sorted(zip(sortedIndices, head), key=lambda x: x[0], reverse=reverse)]

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
                        
    reverse = False
    tail = [h for (_, h) in sorted(zip(sortedIndices, tail), key=lambda x: x[0], reverse=reverse)]

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
            if street.succ:
                return street.succ.intersection
            else:
                return None
        else:
            if street.pred:
                return street.pred.intersection
            else:
                return None
        
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

    # from debug import plt, plotEnd
    # for intersection, street in innerIsects.items():
    #     p = intersection.location
    #     plt.plot(p[0],p[1],'co',markersize=8)
        # for item in street.iterItems():
        #     if isinstance(item, Section):
        #         item.polyline.plot('c',2,'solid',False,999)

    innerStreets = set()
    bundleIsects = set()
    for intersection, innerStreet in innerIsects.items():
        innerStreets.add(innerStreet)
        # Maybe this inner street has inner (minor) intersections
        # bundleIsects.union( innerNodes(innerStreet) )
        lastIsect = lastIsectSeenFrom(intConn.intersection,innerStreet)
        if lastIsect and lastIsect not in innerIsects:
            bundleIsects.add(lastIsect)

    # for intersection in bundleIsects:
    #     p = intersection.location
    #     plt.plot(p[0],p[1],'mo',markersize=8,zorder=800)



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

def canBeMerged(streetGenerator, involvedBundles):
    # When two bundles meet, they can be merged, if there are
    # no inner streets between them at the intersection area.
    bundleList = list(involvedBundles.items())
    (_, streetData0), (_, streetData1) = bundleList

    # Check, if they have all ends common
    ends0 = set( streetData0[0]['bundle'].headLocs if streetData0[0]['type']=='head' else streetData0[0]['bundle'].tailLocs )
    ends1 = set( streetData1[0]['bundle'].headLocs if streetData1[0]['type']=='head' else streetData1[0]['bundle'].tailLocs )
    if ends0 != ends1:
        return False

    # Find the end-points of both streets (see below) and the street
    # at the left border of bundle 0.
    a = streetData0[0]['end']
    c = streetData0[-1]['end']
    street0 = streetData0[0]['street']
    # Find the street of bundle 1, that meets the same endpoint.
    street1 = next(filter(lambda x: x['end']==a,streetData1))['street']
    # determine the succession of these streets through both bundles
    arriving, leaving = (street0,street1) if street0.dst == street1.src else (street1,street0)

    # Check, if the inner streets would be on the left or the right of street0.
    #
    #     checkLeft: True                             checkLeft: False
    #  ============c============  street1          ============a===b========> street0
    #  bundle0     |     bundle1             or    bundle0     |      bundle1 
    #  ============a===b========> street0          ============c============  street1

    b = leaving.head.polyline[1]
    checkLeft = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) > 0.

    # if a in streetGenerator.majorIntersections:
    intersection = streetGenerator.majorIntersections[a]
    # elif a in streetGenerator.minorIntersections:
    #     intersection = streetGenerator.minorIntersections[a]

    initialStreet, followStreet = (leaving, arriving) if checkLeft else (arriving, leaving)

    # Find the initial street
    for conn in IntConnector.iterate_from(intersection.startConnector):
        if conn.item == initialStreet:
            break

    # The circular list of connectors of this intersection is
    # ordered counter-clockwise. When we start with the initial street,
    # all the streets, that are not the followStreet street, inner streets.
    isMergable = True
    for conn in IntConnector.iterate_from(conn.succ):
        if conn.item == followStreet:
            break
        isMergable = False
        break

    return isMergable

def mergeBundles(streetGenerator,involvedBundles):
    # TODO: bundle attribute of inner streets

    # data for bundle 0 from left to right, relative to bundle
    bundleList = list(involvedBundles.items())
    (bundle0, streetData0), (bundle1, streetData1) = bundleList

    # Find the end-points of both sides of the bundle 0. 
    end0left = streetData0[0]['end']
    street0 = streetData0[0]['street']
    intersection0 = streetGenerator.majorIntersections[end0left]

    # Find the street of bundle 1, that meets at the left endpoint of bundle 0
    street1 = next(filter(lambda x: x['end']==end0left,streetData1))['street']
    # determine the succession of these streets through both bundles
    arriving0, leaving0 = (street0,street1) if street0.dst == street1.src else (street1,street0)

    end0right = streetData0[-1]['end']
    street0 = streetData0[-1]['street']
    intersection1 = streetGenerator.majorIntersections[end0right]

    # Find the street of bundle 1, that meets at the right endpoint of bundle 0
    street1 = next(filter(lambda x: x['end']==end0right,streetData1))['street']
    # determine the succession of these streets through both bundles
    arriving1, leaving1 = (street0,street1) if street0.dst == street1.src else (street1,street0)

    # merge these adajacent streets by pseudo minors
    mergedStreets = []

    longStreet = Street(arriving0.src,arriving0.dst)
    longStreet.insertStreetEnd(arriving0)
    longStreet.pred = arriving0.pred
    if longStreet.pred:
        longStreet.pred.item = longStreet
    makePseudoMinor(streetGenerator, intersection0, arriving0, leaving0)
    longStreet.insertEnd(intersection0)
    longStreet.insertStreetEnd(leaving0)
    if longStreet.succ:
        longStreet.succ.item = longStreet
    mergedStreets.append(longStreet)

    longStreet = Street(arriving1.src,arriving1.dst)
    longStreet.insertStreetEnd(arriving1)
    longStreet.pred = arriving1.pred
    if longStreet.pred:
        longStreet.pred.item = longStreet
    makePseudoMinor(streetGenerator, intersection1, arriving1, leaving1)
    longStreet.insertEnd(intersection1)
    longStreet.insertStreetEnd(leaving1)
    if longStreet.succ:
        longStreet.succ.item = longStreet
    mergedStreets.append(longStreet)

    head, tail = orderHeadTail(mergedStreets)

    mergedBundle = Bundle()
    for item in head:
        street = item['street']
        street.bundle = mergedBundle
        mergedBundle.streetsHead.append(street)
        mergedBundle.headLocs.append(item['firstVert'])
    for item in tail:
        street = item['street']
        street.bundle = mergedBundle
        mergedBundle.streetsTail.append(street)
        mergedBundle.tailLocs.append(item['firstVert'])

    streetGenerator.bundles[mergedBundle.id] = mergedBundle

    # Bookkeeping:
    # remove old bundles
    for bundle,_ in involvedBundles.items():
        if bundle.id in streetGenerator.bundles:
            del streetGenerator.bundles[bundle.id]
    # Redirect inner streets, if any
    for _,street in streetGenerator.streets.items():
        if street.bundle in [bundle0, bundle1]:
            street.bundle = mergedBundle


def intersectBundles(streetGenerator,involvedBundles):
    if len(involvedBundles) == 2:
        twoBundleIntersection(streetGenerator,involvedBundles)
    else:
        multiBundleIntersection(streetGenerator,involvedBundles)

def twoBundleIntersection(streetGenerator,involvedBundles):
    bundleStreets = set()
    ends = set()
    for _,data in involvedBundles.items():
        bundleStreets = bundleStreets.union( set(item['street'] for item in data) )
        ends = ends.union( set(item['end'] for item in data) )

    # find streets, that do not belong to the bundle
    externalStreets = set()
    for end in ends:
        if end in streetGenerator.majorIntersections:
            intersection = streetGenerator.majorIntersections[end]
            for intSec in intersection:
                if intSec.item not in bundleStreets:
                    externalStreets.add(intSec.item)

    # Now, we have to find, which external streets are between the
    # outer borders of the bundles. These must be removed.
    bundleList = list(involvedBundles.items())
    (bundle0, data0), (bundle1, data1) = bundleList

    # Find the border streets and their end points for the first bundle.
    type = data0[0]['type']
    leftStreet = data0[0]['street'] if type=='head' else data0[-1]['street']
    leftEnd = data0[0]['end'] if type=='head' else data0[-1]['end']
    leftFirst = leftStreet.head.polyline[1] if leftStreet.src==leftEnd else leftStreet.tail.polyline[-2]
    rightStreet = data0[-1]['street'] if type=='head' else data0[0]['street']
    rightEnd = data0[-1]['end'] if type=='head' else data0[0]['end']
    rightFirst = rightStreet.head.polyline[1] if rightStreet.src==rightEnd else rightStreet.tail.polyline[-2]

    # Now, we are able to determine, if a point <o> is between the border streets. 
    # It must be right of a-b and left of c-d, if
    # a: leftEnd, b: leftFirst, c: rightEnd, d: rightFirst
    #
    #  ============a===b========  leftStreet          ============a============> leftStreet
    #              |                                              |         
    #              |         bundle            and                |        bundle
    #              |     o                                        |     o 
    #  ============c============> rightStreet         ============c===d======== rightStreet
    #                    o_isRightOfLeft                                o_isLeftOfRight

    # Let's check that for the endpoints of the external Streets.
    a = leftEnd
    b = leftFirst
    c = rightEnd
    d = rightFirst
    epsilon = 1.e-5
    streetsLeftOfBundle = []
    streetsRightOfBundle = []
    for street in externalStreets:
        o = street.src if street.dst in ends else street.dst
        o_isRightOfLeft = (b[0] - a[0])*(o[1] - a[1]) - (b[1] - a[1])*(o[0] - a[0]) <=  epsilon
        o_isLeftOfRight = (d[0] - c[0])*(o[1] - c[1]) - (d[1] - c[1])*(o[0] - c[0]) >= -epsilon
        # print( o_isRightOfLeft, o_isLeftOfRight)
        # from debug import plt, plotQualifiedNetwork, randomColor, plotEnd
        # plt.close()
        # plt.plot([a[0],b[0]],[a[1],b[1]],'g',linewidth=4,zorder=950)
        # plt.plot([c[0],d[0]],[c[1],d[1]],'r',linewidth=4,zorder=950)
        # plt.plot(a[0],a[1],'go',zorder=950)
        # plt.plot(c[0],c[1],'ro',zorder=950)
        # plt.plot(o[0],o[1],'kX',markersize=12,zorder=950)
        # plotEnd()
        if o_isRightOfLeft and o_isLeftOfRight:
            # The street is inside the bundle, remove it
            del streetGenerator.streets[street.id]
        elif o_isLeftOfRight:
            # The street is left of the bundle
            streetsLeftOfBundle.append(street)
        elif o_isRightOfLeft:
            # The street is right of the bundle
            streetsRightOfBundle.append(street)

    # Now, we are able to construct the intersection
    location = sum(ends,Vector((0,0)))/len(ends)
    intersection = Intersection(location)
    intersection.connectsBundles = True

    # The bundle0 makes the start "upwards"
    connector = IntConnector(intersection)
    connector.item = bundle0
    connector.leaving = type=='head'
    if connector.leaving:
        bundle0.pred = connector
    else:
        bundle0.succ = connector
    intersection.insertConnector(connector)

    # To be counter-clockwise, we insert now the left streets.
    # TODO: This works currently only for one street. If more
    # streets, they will have to be sorted counter-clockwise.
    if streetsLeftOfBundle:
        if len(streetsLeftOfBundle)>1:
            print('This works currently only for one side street !!!')
        street = streetsLeftOfBundle[0]
        connector = IntConnector(intersection)
        connector.item = street
        connector.leaving = street.src in ends
        if connector.leaving:
            street.pred = connector
        else:
            street.succ = connector
        intersection.insertConnector(connector)

    # The bundle1 looks now "downwards"
    connector = IntConnector(intersection)
    connector.item = bundle1
    type = data1[0]['type']
    connector.leaving = type=='head'
    if connector.leaving:
        bundle1.pred = connector
    else:
        bundle1.succ = connector
    intersection.insertConnector(connector)


    # Finally, we insert the right streets.
    # TODO: This works currently only for one street. If more
    # streets, they will have to be sorted counter-clockwise.
    if streetsRightOfBundle:
        if len(streetsRightOfBundle)>1:
            print('This works currently only for one side street !!!')
        street = streetsRightOfBundle[0]
        connector = IntConnector(intersection)
        connector.item = street
        connector.leaving = street.src in ends
        if connector.leaving:
            street.pred = connector
        else:
            street.succ = connector
        intersection.insertConnector(connector)

    streetGenerator.majorIntersections[intersection.location] = intersection

    # Remove intersections???
    for end in ends:
        if end in streetGenerator.majorIntersections:
            del streetGenerator.majorIntersections[end]

def multiBundleIntersection(streetGenerator,involvedBundles):
    ends = set()
    intersectingBundles = dict()
    for bundle,data in involvedBundles.items():
        types = set(item['type'] for item in data)
        if len(types)>1:    
            # Head and tail of this bundle sre completely in intersection.
            # Remove it, including its streets
            for street in bundle.streetsHead:
                if street.id in streetGenerator.streets:
                    del streetGenerator.streets[street.id]
            for street in bundle.streetsTail:
                if street.id in streetGenerator.streets:
                    del streetGenerator.streets[street.id]
            if bundle.id in streetGenerator.bundles:
                del streetGenerator.bundles[bundle.id]
        else:   # Bundle leaves intersection
            ends = ends.union( set(item['end'] for item in data) )
            intersectingBundles[bundle] = data
    pass

    # Order these ends counter-clockwise around center of gravity.
    # The connectors of their intersection are order like this.
    location = sum(ends,Vector((0,0)))/len(ends)
    location.freeze()
    ends = sorted(ends, key=lambda x: _pseudoangle(x-location))

    for end in ends:
        if end in streetGenerator.majorIntersections:
            del streetGenerator.majorIntersections[end]
        if end in streetGenerator.minorIntersections:
            del streetGenerator.minorIntersections[end]

    # Create the intersection
    intersection = Intersection(location)
    intersection.connectsBundles = True
    # TODO: This works only for two street bundles !!!
    for e0,e1 in cyclePairs(ends):
        for bundle, data in intersectingBundles.items():
            type = data[0]['type']
            bundleEnds = bundle.headLocs if type=='head' else bundle.tailLocs
            if e0 in bundleEnds and e1 in bundleEnds:
                connector = IntConnector(intersection)
                connector.item = bundle
                connector.leaving = type=='head'
                if connector.leaving:
                    bundle.pred = connector
                else:
                    bundle.succ = connector
                intersection.insertConnector(connector)

    streetGenerator.majorIntersections[intersection.location] = intersection


def endBundleIntersection(streetGenerator, bundle):
    if not bundle.pred:
        externalStreets = set()
        for end in bundle.headLocs:
            if end not in streetGenerator.majorIntersections:
                pass
            else:
                intersection = streetGenerator.majorIntersections[end]
                for intSec in intersection:
                    if intSec.item not in bundle.streetsHead:
                        externalStreets.add(intSec.item)
        if not externalStreets:
            bundle.pred = None  # Void end of bundle
        else:
            # We have an end-intersection at the head of this bundle
            location = sum(bundle.headLocs,Vector((0,0)))/len(bundle.headLocs)
            location.freeze()
            intersection = Intersection(location)
            intersection.connectsBundles = True

            # The bundle makes the start "upwards"
            connector = IntConnector(intersection)
            connector.item = bundle
            connector.leaving = True
            bundle.pred = connector
            intersection.insertConnector(connector)

            # A counter-clockwise rotation of the intersections starts
            # with the leftmost end in the bundle.
            for end in bundle.headLocs:
                if end in streetGenerator.majorIntersections:
                    endIsect = streetGenerator.majorIntersections[end]
                    # The iterator of Intersection runs counter-clockwise
                    for intSec in endIsect:
                        if intSec.item not in bundle.streetsHead:
                            connector = IntConnector(intersection)
                            connector.item = intSec.item
                            connector.leaving = intSec.item.src==end
                            if connector.leaving:
                                intSec.item.pred = connector
                            else:
                                intSec.item.succ = connector
                            intersection.insertConnector(connector)

            # Remove intersections???
            for end in bundle.headLocs:
                if end in streetGenerator.majorIntersections:
                    del streetGenerator.majorIntersections[end]

            streetGenerator.majorIntersections[intersection.location] = intersection


    if not bundle.succ:
        externalStreets = set()
        for end in bundle.tailLocs:
            if end not in streetGenerator.majorIntersections:
                pass
            else:
                intersection = streetGenerator.majorIntersections[end]
                for intSec in intersection:
                    if intSec.item not in bundle.streetsTail:
                        externalStreets.add(intSec.item)
        if not externalStreets:
            bundle.succ = None  # Void end of bundle
        else:
            # We have an end-intersection at the head of this bundle
            location = sum(bundle.tailLocs,Vector((0,0)))/len(bundle.tailLocs)
            location.freeze()
            intersection = Intersection(location)
            intersection.connectsBundles = True

            # The bundle makes the start "upwards"
            connector = IntConnector(intersection)
            connector.item = bundle
            connector.leaving = False
            bundle.succ = connector
            intersection.insertConnector(connector)

            # A counter-clockwise rotation of the intersections starts
            # with the leftmost end in the bundle. Because this is a 
            # bundle tail, this is the rightmost in the bundle.
            # Therefore reverse the locations.
            for end in bundle.tailLocs[::-1]:
                if end in streetGenerator.majorIntersections:
                    endIsect = streetGenerator.majorIntersections[end]
                    # The iterator of Intersection runs counter-clockwise
                    for intSec in endIsect:
                        if intSec.item not in bundle.streetsHead:
                            connector = IntConnector(intersection)
                            connector.item = intSec.item
                            connector.leaving = intSec.item.src==end
                            if connector.leaving:
                                intSec.item.pred = connector
                            else:
                                intSec.item.succ = connector
                            intersection.insertConnector(connector)

            # Remove intersections???
            for end in bundle.tailLocs:
                if end in streetGenerator.majorIntersections:
                    del streetGenerator.majorIntersections[end]

            streetGenerator.majorIntersections[intersection.location] = intersection
