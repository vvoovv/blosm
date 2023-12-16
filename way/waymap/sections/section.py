from lib.CompGeom.PolyLine import PolyLine

class Section(object):
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    def __init__(self,net_section,network):
        self.id = Section.ID
        Section.ID += 1

        self.category = net_section.category
        self.tags = net_section.tags

        self._src = net_section.s
        self._dst = net_section.t

        self.polyline = PolyLine(net_section.path)
        self.isLoop = self._src == self._dst

        self.isOneWay = None
        self.lanePatterns = None
        self.totalLanes = None
        self.forwardLanes = None
        self.backwardLanes = None
        self.bothLanes = None

    @property
    def src(self):
        return self._src
    
    @property
    def dst(self):
        return self._dst
