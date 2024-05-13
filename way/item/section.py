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


    def insertAfter(self, length, item):
        # Shortens the section to the length <length> and appends <item> after it. <length> is
        # the physical length in meters. After the split the section is followed by <item>. The
        # method returns True, if item is successfully inserted and False else.
        # For now, the method returns False, when <length> is larger than the length of the section.
        # <item> is inserted into the linked list by:
        # section.succ.pred = item    # if pred exists
        # item.succ = section.succ
        # section.succ = item
        # item. pred = section

        # Shorten section (self)
        t = self.polyline.d2t(length)
        endT = len(self.polyline)-1
        if 0. <= t < endT:
            self.polyline = self.polyline.trimmed(0.,t)
            remainingPolyline = self.polyline.trimmed(t,endT)
            self.trimT = len(self.polyline)-1
            self.centerline = self.polyline[::]
            self._dst = self.polyline[-1]

            # Add remaining centerline to item
            item.centerline = remainingPolyline[::]

            # insert item in linked list
            if hasattr(self.succ, 'pred'):  # check, maybe its an intersection
                self.succ.pred = item
            item.succ = self.succ
            self.succ = item
            item.pred = self
            return True
        else:
            return False

    def insertBefore(self, length, item):
        # Shortens the section so that it starts after the length <length> and remains until its end.
        # <length> is the physical length in meters. After the split the item is followed by the
        # remaining section. The method returns True, if <item> is successfully inserted and False else.
        # For now, the method returns False, when <length> is larger than the length of the section.
        # <item> is inserted into the linked list by:
        # section.pred.succ = item  # if succ exists
        # item.pred = section.pred
        # item.succ = section
        # section.pred = item

        # Shorten section (self)
        t = self.polyline.d2t(length)
        lenOld = len(self.polyline)-1
        if 0. <= t < lenOld:
            self.polyline = self.polyline.trimmed(t,lenOld)
            remainingPolyline = self.polyline.trimmed(0,t)
            self.trimT = len(self.polyline)-1
            self.centerline = self.polyline[::]
            self._src = self.polyline[0]

            # Add remaining centerline to item
            item.centerline = remainingPolyline[::]

            # insert item in linked list
            if hasattr(self.pred, 'succ'):  # check, maybe its an intersection
                self.pred.succ = item
            item.pred = self.pred
            item.succ = self
            self.pred = item
            return True
        else:
            return False

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