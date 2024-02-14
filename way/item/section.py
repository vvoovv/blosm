from .item import Item
from way.way_properties import getWidth


class Section(Item):
    
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self, net_section, polyline, network):
        super().__init__()
        self.id = Section.ID
        Section.ID += 1

        self.category = net_section.category    # the category of the related OSM-way
        self.centerline = polyline[::]          # a Python list of coordinates
        self.offset = 0.                        # an offset from the centerline
        self.width = 0.                         # the width of the section

        self.oneway = None
        self.forwardLanes = None
        self.backwardLanes = None
        self.bothLanes = None
        self.totalLanes = None
        self.laneWidth = 0.

        self.valid = True

        # Values of the attributes pred and succ can be None in the case of a dead-end, 
        # or an instance of Intersection, PartialIntersection, Crosswalk, TransitionSideLane,
        # TransitionSymLane, SplitMerge, or PtStop.
        self.pred = None
        self.succ = None

        # The following attributes are used only internally in StreetGenerator
        self.tags = net_section.tags

        self._src = polyline[0]
        self._dst = polyline[-1]

        self.polyline = polyline
        self.isLoop = self._src == self._dst

        self.lanePatterns = None

        self.trimS = 0.                      # trim factor for start
        self.trimT = len(self.polyline)-1    # trim factor for target


    def split(self, nodeIndex, item, itemLength):
        # The method splits the section at the <nodeIndex> and inserts the <item> which has
        # the length <itemLength>. The attributes centerline, pred, succ of the items in question
        # are changed or set accordingly.
        pass    # to be implemented

    def trimStart(self, item, itemLength):
        # The method trims the section at the start and inserts the <item> at the start of the section.
        # The <item> has the length <itemLength>. The attributes centerline, pred, succ of the items in
        # question are changed or set accordingly.
        pass    # to be implemented

    def trimEnd(self, item, itemLength):
        # The method trims the section at the end and inserts the <item> at the end of the section.
        # The <item> has the length <itemLength>. The attributes centerline, pred, succ of the items in
        # question are changed or set accordingly.
        pass    # to be implemented

    @property
    def src(self):
        return self._src
    
    @property
    def dst(self):
        return self._dst
    
    def setSectionAttributes(self, oneway, fwdPattern, bwdPattern, bothLanes, props):
        self.oneway = oneway
        self.lanePatterns = (fwdPattern,bwdPattern)
        self.totalLanes = len(fwdPattern) + len(bwdPattern) + bothLanes
        self.forwardLanes = len(fwdPattern)
        self.backwardLanes = len(bwdPattern)
        self.bothLanes = bothLanes

        width = getWidth(self.tags)
        self.laneWidth = width/self.totalLanes if width else props['laneWidth']
        self.width = self.totalLanes * self.laneWidth
        self.forwardWidth =  self.forwardLanes * self.laneWidth + \
                                self.bothLanes * self.laneWidth/2.
        self.backwardWidth = self.backwardLanes * self.laneWidth + \
                                self.bothLanes * self.laneWidth/2.
    
    def getCategory(self):
        return self.category
    
    def getName(self):
        return self.tags.get("name")