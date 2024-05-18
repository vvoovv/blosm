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

        # A reference to the incoming Section before the turn lanes
        self.pred = incoming

        # A reference to the outgoing Section with the turn lanes
        self.succ = outgoing

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
        self.totalLanesIncreased = self.succ.totalLanes > self.pred.totalLanes

        self.createSideLane()

    @property
    def location(self):
        return self._location

    def createSideLane(self):
        # Way width and offset correction for seamless connections according to
        # https://github.com/prochitecture/blosm/issues/57#issuecomment-1544025403
        widePixels = self.succ.totalLanes * 220.0 + 8 * (self.succ.totalLanes - 1)
        smallPixels = self.pred.totalLanes * 220.0 + 8 * (self.succ.totalLanes - 2)
        factor = widePixels / smallPixels
        # Fix ratio by changing wider way
        self.succ.width = factor * self.pred.width
        self.succ.laneWidth = self.succ.width / self.succ.totalLanes

        # Create direction indicators, positive, if start of Section at location
        signOfTurn = signOfPre = 1 if self.pred.src == self.location else -1
        signOfTurn = signOfTurn = 1 if self.succ.src == self.location else -1
        self.directions = (signOfPre, signOfTurn)

        # Determine turn types and offset
        if self.pred.oneway:  # then also outgoing.oneway is True
            fwdL, fwdR = turnsFromPatterns(
                self.succ.lanePatterns[0], self.pred.lanePatterns[0]
            )
            isNotMerge = fwdL not in ["l", "r"]
            self.laneL = bool(fwdL) if isNotMerge else not bool(fwdL)
            self.laneR = bool(fwdR) if isNotMerge else not bool(fwdR)
            self.succ.fwdLaneR = fwdR
            self.succ.fwdLaneL = fwdL
            laneDir = 1 if self.laneR else -1
            if self.laneL or self.laneR:
                # The reference is the centerline of the section with the turn lane (which is
                # always the outgoing section of the TransitionSideLane, even if the centerline
                # points inwards). If the offset is negative, the curve is offset to the left
                # relative to the supplied centerline. Otherwise, the curve is offset to the right.
                # signOfTurn is positive, if in outgoing direction of the turn way
                self.succ.offset = (self.succ.width-self.pred.width) / 2. * signOfTurn * laneDir
        else:  # two-ways
            # TODO case of merging lanes, as above?
            fwdL, fwdR = turnsFromPatterns(
                self.succ.lanePatterns[0], self.pred.lanePatterns[0]
            )
            bwdL, bwdR = turnsFromPatterns(
                self.succ.lanePatterns[1], self.pred.lanePatterns[1]
            )
            self.laneL = bool(fwdL) or bool(bwdL)
            self.laneR = bool(fwdR) or bool(bwdR)
            self.succ.fwdLaneR = fwdR
            self.succ.fwdLaneL = fwdL
            self.succ.bwdLaneR = bwdR
            self.succ.bwdLaneL = bwdL
            if self.laneL or self.laneR:
                # The reference is the centerline of the section with the turn lane (which is
                # always the outgoing section of the TransitionSideLane, even if the centerline
                # points inwards). If the offset is negative, the curve is offset to the left
                # relative to the supplied centerline. Otherwise, the curve is offset to the right.
                # signOfTurn is positive, if in outgoing direction of the turn way
                nrTurnLanesFwd = max(len(fwdL), len(fwdR))
                # nrTurnLanesBwd = max( len(bwdL), len(bwdR))
                self.succ.offset = (
                          (self.succ.width-self.pred.width) / 2. * signOfTurn
                    if self.laneR else
                        - (self.succ.width-self.pred.width) / 2. * signOfTurn
                 )

    def splitAffectedSection(self):
        length = self.length = self.getStyleBlockAttr("length")
        
        if self.totalLanesIncreased:
            self.succ.insertBefore(length, self, updateNeighbors=False)
        else:
            self.pred.insertAfter(length, self, updateNeighbors=False)
    
    def getClass(self):
        return (self.pred if self.totalLanesIncreased else self.succ).getStyleBlockAttr("cl")