from collections import defaultdict

from .item import Item
from way.item.street import Street
from way.item.section import Section
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
    intersection.isMajor = False

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
    for street in streetGroup:
        streetEnds[street.src].append(street)
        streetEnds[street.dst].append(street)
        intersections[street.src] = street.pred.intersection if street.pred else None
        intersections[street.dst] = street.succ.intersection if street.succ else None
    return streetEnds, intersections

# see https://github.com/prochitecture/blosm/issues/104#issuecomment-2322836476
# Major intersections the street group of a bundle, with only one side street,
# are merged into a long street, similar to minor intersections.
def mergePseudoMinors(streetGenerator, streetGroup, streetEnds, intersections):

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

