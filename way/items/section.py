from lib.CompGeom.PolyLine import PolyLine

class Section(object):
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    def __init__(self,net_section,polyline,network):
        self.id = Section.ID
        Section.ID += 1

        self.category = net_section.category
        self.tags = net_section.tags

        self._src = polyline[0]
        self._dst = polyline[-1]

        self.polyline = polyline
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
    
    def setlaneParams(self, isOneWay, fwdPattern, bwdPattern, bothLanes):
        self.isOneWay = isOneWay
        self.lanePatterns = (fwdPattern,bwdPattern)
        self.totalLanes = len(fwdPattern) + len(bwdPattern) + bothLanes
        self.forwardLanes = len(fwdPattern)
        self.backwardLanes = len(bwdPattern)
        self.bothLanes = bothLanes
