from lib.CompGeom.PolyLine import PolyLine

class WaySection():
    # _frozen = False
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    def __init__(self,net_section,network):
        self.id = WaySection.ID
        WaySection.ID += 1
        self.network = network
        self.originalSection = net_section
        self.category = self.originalSection.category
        self.tags = self.originalSection.tags
        self.polyline = PolyLine(net_section.path)
        self.isLoop = net_section.s == net_section.t
        self._sV = None  # vector of first segment (cached when computed)
        self._tV = None  # vector of last segment (cached when computed)
        self.trimS = 0.                      # trim factor for start
        self.trimT = len(self.polyline)-1    # trim factor for target
        self._isValid = True

        # Lane data, if any. Filled up within createWaySections() of StreetGenerator.
        self.isOneWay = False       # True if one-way road.
        self.lanePatterns = None    # (fwdPattern,bwdPattern), String representation of lane types, fwd and bwd.
        self.totalLanes = None      # Total number of lanes, includes bothLanes
        self.forwardLanes = None    # Number of forward lanes
        self.backwardLanes = None   # Number of backward lanes
        self.bothLanes = None       # Number of lanes that allow a turn available in both directions (tag lanes:both_ways)
        self.laneWidth = None       # Width of a single lane
        self.width = None           # Width of all lanes
        self.forwardWidth = None    # Width of forward lanes, including half of both (tag lanes:both_ways)
        self.backwardWidth = None   # Width of backward lanes, including half of both (tag lanes:both_ways)

        self.fwdLaneR = False
        self.fwdLaneL = False
        self.bwdLaneR = False
        self.bwdLaneL = False

        # The reference for offset is the centerline of the section with the turn lane (which is
        # always the outgoing section of the <TransitionSideLane>, even if the centerline
        # points inwards). If the offset is negative, the curve is offset to the left
        # relative to the supplied centerline. Otherwise, the curve is offset to the right.   
        # signOfTurn is positive, if in outgoing direction of the turn way          
        self.offset = 0.        
        self.signOfTurn = 1    


        # Trim values in a cluster
        self.trimStart = 0.
        self.trimEnd = 0.
        # self._frozen = True

    # def __setattr__(self, attr, value):
    #    if getattr(self, "_frozen") and not hasattr(self,attr):
    #         raise AttributeError("Trying to set attribute " + attr + " of a frozen instance")
    #    return super().__setattr__(attr, value) 

    @property
    def leftW(self):
        return self.width/2.# + 2.*self.offset
 
    @property
    def rightW(self):
        return - self.width/2.# - 2.*self.offset)
 
    @property
    def isClipped(self):
        return self._isClipped

    @isClipped.setter
    def isClipped(self,val):
        self._isClipped = val

    @property
    def isValid(self):
        return self._isValid

    @isValid.setter
    def isValid(self,val):
        self._isValid = val

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
        return [v for v in self.polyline[::-1]]

    