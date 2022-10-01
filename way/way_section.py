from lib.CompGeom.PolyLine import PolyLine
from way.way_properties import estimateWayWidth, getLanes, isOneWay


class WaySection():
    ID = 0
    def __init__(self,net_section):
        self.originalSection = net_section
        self.rightWidth = self.leftWidth = estimateWayWidth(net_section.category,net_section.tags)/2.
        self.polyline = PolyLine(net_section.path)
        self.isLoop = net_section.s == net_section.t
        self._sV = None  # vector of first segment
        self._tV = None  # vector of last segment
        self.trimS = 0.                      # trim factor for start
        self.trimT = len(self.polyline)-1    # trim factor for target
        self.id = WaySection.ID
        self.turnParams = None  # parameters for turning lanes
        self.isValid = True
        WaySection.ID += 1
        self.isOneWay = isOneWay(self.originalSection.tags)
        nrOfLanes = getLanes(self.originalSection.tags)
        if nrOfLanes:
            if self.isOneWay:
                self.nrLeftLanes = 0
                self.nrRightLanes = nrOfLanes
            else:
                self.nrLeftLanes = nrOfLanes//2
                self.nrRightLanes = nrOfLanes//2
        else:
            self.nrLeftLanes = 1
            self.nrRightLanes = 1
        self.processTurnLanes()

    @property
    def sV(self):
        if self._sV is None:
            self._sV = self.polyline.verts[1] - self.polyline.verts[0]
        return self._sV

    @property
    def tV(self):
        if self._tV is None:
            self._tV = self.polyline.verts[-2] - self.polyline.verts[-1]
        return self._tV

    def fwd(self):
        return [v for v in self.polyline]

    def rev(self):
        return [v for v in self.polyline.reversed()]

    def processTurnLanes(self):
        width = estimateWayWidth(self.originalSection.category,self.originalSection.tags)
        tags = self.originalSection.tags
        if 'turn:lanes' in tags:
            nrOfLanes = getLanes(tags)
            if nrOfLanes:
                laneWidth = width/nrOfLanes
                laneDescs = tags['turn:lanes'].split('|')
                leftTurnLanes = sum(1 for tag in laneDescs if 'left' in tag)
                rightTurnLanes = sum(1 for tag in laneDescs if 'right' in tag)
                if leftTurnLanes == rightTurnLanes:
                    self.leftWidth = width/2.
                    self.rightWidth = width/2.
                else:
                    halfSymLaneWidth = laneWidth*(nrOfLanes - leftTurnLanes - rightTurnLanes)/2
                    self.leftWidth = halfSymLaneWidth + leftTurnLanes * laneWidth
                    self.rightWidth = halfSymLaneWidth + rightTurnLanes * laneWidth
                    if not self.isOneWay:
                        self.nrLeftLanes += leftTurnLanes - rightTurnLanes
                        self.nrRightLanes += rightTurnLanes - leftTurnLanes
            else:
                self.leftWidth = width/2.
                self.rightWidth = width/2.
        else:
            self.leftWidth = width/2.
            self.rightWidth = width/2.

