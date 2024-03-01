from .item import Item
from way.way_properties import turnsFromPatterns


class SideLane(Item):
    # This class describes a transition between two Sections, where one of them has 
    # turn lanes additionally to the others. At the transition, the two Sections are not trimmed.

    ID = 0

    def __init__(self, location, incoming, outgoing):
        super().__init__()
        self.id = SideLane.ID
        SideLane.ID += 1

        self._location = location
        self.pred = None
        self.succ = None

        # A reference to the incoming Section before the turn lanes
        self.incoming = incoming

        # A reference to the outgoing Section with the turn lanes
        self.outgoing = outgoing

        # A tuple of direction indicators of the Sections, that connect to this transition.
        # The first indicator refers to the incoming Section and the second to the outgoing
        # Section. The indicator is positive, when the start of the Section connects to the
        # transition in the middle between them, and negative, when the Section ends here.
        self.directions = None

        # True, if there are turn lanes to the left from the narrower way-section.
        self.laneL = False

        # True, if there are turn lanes to the right from the narrower way-section.
        self.laneR = False

        # self.turnSection.totalLanes > self.preTurnSection.totalLanes
        self.totalLanesIncreased = self.outgoing.totalLanes > self.incoming.totalLanes

        self.createSideLane()

    @property
    def location(self):
        return self._location

    def createSideLane(self):
        # Way width and offset correction for seamless connections according to
        # https://github.com/prochitecture/blosm/issues/57#issuecomment-1544025403
        widePixels = self.outgoing.totalLanes * 220.0 + 8 * (self.outgoing.totalLanes - 1)
        smallPixels = self.incoming.totalLanes * 220.0 + 8 * (self.outgoing.totalLanes - 2)
        factor = widePixels / smallPixels
        # Fix ratio by changing wider way
        self.outgoing.width = factor * self.incoming.width
        self.outgoing.laneWidth = self.outgoing.width / self.outgoing.totalLanes

        # Create direction indicators, positive, if start of Section at location
        signOfTurn = signOfPre = 1 if self.incoming.src == self.location else -1
        signOfTurn = signOfTurn = 1 if self.outgoing.src == self.location else -1
        self.directions = (signOfPre, signOfTurn)

        # Determine turn types and offset
        if self.incoming.oneway:  # then also outgoing.oneway is True
            fwdL, fwdR = turnsFromPatterns(
                self.outgoing.lanePatterns[0], self.incoming.lanePatterns[0]
            )
            isNotMerge = fwdL not in ["l", "r"]
            self.laneL = bool(fwdL) if isNotMerge else not bool(fwdL)
            self.laneR = bool(fwdR) if isNotMerge else not bool(fwdR)
            self.outgoing.fwdLaneR = fwdR
            self.outgoing.fwdLaneL = fwdL
            laneDir = 1 if self.laneR else -1
            if self.laneL or self.laneR:
                # The reference is the centerline of the section with the turn lane (which is
                # always the outgoing section of the TransitionSideLane, even if the centerline
                # points inwards). If the offset is negative, the curve is offset to the left
                # relative to the supplied centerline. Otherwise, the curve is offset to the right.
                # signOfTurn is positive, if in outgoing direction of the turn way
                nrTurnLanesFwd = max(len(fwdL), len(fwdR))
                self.outgoing.offset = (
                    self.outgoing.laneWidth / 2.0 * signOfTurn * laneDir * nrTurnLanesFwd
                )
        else:  # two-ways
            # TODO case of merging lanes, as above?
            fwdL, fwdR = turnsFromPatterns(
                self.outgoing.lanePatterns[0], self.incoming.lanePatterns[0]
            )
            bwdL, bwdR = turnsFromPatterns(
                self.outgoing.lanePatterns[1], self.incoming.lanePatterns[1]
            )
            self.laneL = bool(fwdL) or bool(bwdL)
            self.laneR = bool(fwdR) or bool(bwdR)
            self.outgoing.fwdLaneR = fwdR
            self.outgoing.fwdLaneL = fwdL
            self.outgoing.bwdLaneR = bwdR
            self.outgoing.bwdLaneL = bwdL
            if self.laneL or self.laneR:
                # The reference is the centerline of the section with the turn lane (which is
                # always the outgoing section of the TransitionSideLane, even if the centerline
                # points inwards). If the offset is negative, the curve is offset to the left
                # relative to the supplied centerline. Otherwise, the curve is offset to the right.
                # signOfTurn is positive, if in outgoing direction of the turn way
                nrTurnLanesFwd = max(len(fwdL), len(fwdR))
                # nrTurnLanesBwd = max( len(bwdL), len(bwdR))
                self.outgoing.offset = (
                    self.outgoing.laneWidth / 2.0 * signOfTurn * nrTurnLanesFwd
                    if self.laneR
                    else -self.outgoing.laneWidth / 2.0 * signOfTurn * nrTurnLanesFwd
                )
