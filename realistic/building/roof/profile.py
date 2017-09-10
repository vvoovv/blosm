import math
from . import RoofRealistic
from building.roof.profile import RoofProfile


class RoofProfileRealistic(RoofRealistic, RoofProfile):
    
    def __init__(self, data):
        super().__init__(data)
        self.texCoords = []
        # mapping between the indices
        self.indicesMap = {}
        # Mapping between a roof polygon (or face) and the related slot;
        # indices in self.roofToSlot exactly correspond to the indices in <self.roofIndices>
        self.roofFaceToSlot = []
        
        slots = self.slots
        p = self.profile
        self.dx_2 = tuple(
            (slots[i+1].x-slots[i].x)*(slots[i+1].x-slots[i].x)
            for i in range(self.lastProfileIndex)
        )
        self.dy_2 = tuple(
            (p[i+1][1]-p[i][1])*(p[i+1][1]-p[i][1])
            for i in range(self.lastProfileIndex)
        )
        # An element of <self.slopes> can take 3 possible value:
        # True (positive slope of a profile part)
        # False (negative slope of a profile part)
        # None (flat profile part)
        self.slopes = tuple(
            True if p[i+1][1]>p[i][1] else
            (False if p[i+1][1]<p[i][1] else None)
            for i in range(self.lastProfileIndex)
        )
        # the lenths of profile parts
        self.partLength = [0. for i in range(self.lastProfileIndex)]
    
    def init(self, element, data, osm, app):
        super().init(element, data, osm, app)
        
        self.texCoords.clear()
        # create slots for polygon vertices in <self.texCoords>
        self.texCoords.extend(None for i in range(self.polygon.n))
        # fill <self.indicesMap> with values
        indicesMap = self.indicesMap
        indicesMap.clear()
        for i,index in enumerate(self.polygon.indices):
            indicesMap[index] = i
        self.roofFaceToSlot.clear()
        # minimum and maximum Y-coordinates in the profile coordinate system
        # for the roof vertices
        self.minY = math.inf
        self.maxY = -math.inf
    
    def _make(self):
        """
        The override of the parent class method
        """
        slots = self.slots
        
        self.polygonWidth_2 = self.polygonWidth * self.polygonWidth
        self.roofHeight_2 = self.roofHeight * self.roofHeight
        for i in range(self.lastProfileIndex):
            self.partLength[i] = self.polygonWidth * (slots[i+1].x-slots[i].x)\
            if self.slopes[i] is None else\
            math.sqrt(self.polygonWidth_2*self.dx_2[i] + self.roofHeight_2 * self.dy_2[i])

    def renderRoofTextured(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        polygon = self.polygon
        uvLayer = bm.loops.layers.uv[0]
        texCoords = self.texCoords
        for indices,slotIndex in zip(self.roofIndices, self.roofFaceToSlot):
            # create a BMesh face for the building roof
            f = bm.faces.new(verts[i] for i in indices)
            for i,roofIndex in enumerate(indices):
                texCoords = self.texCoords[
                    self.indicesMap[roofIndex]\
                    if roofIndex < polygon.indexOffset\
                    else polygon.n + roofIndex - polygon.indexOffset
                ]
                slope = self.slopes[slotIndex]
                #
                # set texture coordinates <u> and <v>
                #
                # <texCoords> is a Python tuple of three elements:
                # <texCoords[0]> indicates if the related roof vertex is located
                #     on the slot;
                # <texCoords[1]> is a slot index if <texCoords[0]> is equal to True;
                # <texCoords[1]> is a coordinate along profile part
                #     if <texCoords[0]> is equal to False;
                # <texCoords[2]> is a coordinate along Y-axis of the profile
                #     coordinate system
                if texCoords[0]:
                    v = 0.\
                    if (slope and texCoords[1] == slotIndex) or\
                    (not slope and texCoords[1] == slotIndex+1) else\
                    self.partLength[slotIndex]
                else:
                    # the related roof vertex isn't located on the slot
                    v = texCoords[1]
                # set texture coordinate <x> depending on the value of <slope>
                u = self.maxY - texCoords[2]\
                    if slope else\
                    texCoords[2] - self.minY
                f.loops[i][uvLayer].uv = (u, v)
            self.mrr.renderRoof(f)

    def getProfiledVert(self, i, roofMinHeight, noWalls):
        """
        The override of the parent class method
        """
        pv = super().getProfiledVert(i, roofMinHeight, noWalls)
        y = pv.y
        texCoords = (
            pv.onSlot,
            pv.index if pv.onSlot else self.getTexCoordAlongProfile(pv),
            y
        )
        if pv.vertIndex < self.polygon.indexOffset:
            self.texCoords[i] = texCoords
        else:
            self.texCoords.append(texCoords)
        # update <self.minY> and <self.maxY> if necessary
        if y < self.minY:
            self.minY = y
        elif y > self.maxY:
            self.maxY = y
        return pv
    
    def getTexCoordAlongProfile(self, pv):
        slots = self.slots
        p = self.profile
        slope = self.slopes[pv.index]
        if slope:
            dx = pv.x - slots[pv.index].x
            dh = pv.h - p[pv.index][1]
            texCoord = math.sqrt(self.polygonWidth_2*dx*dx + self.roofHeight_2*dh*dh)
        elif slope is False:
            dx = slots[pv.index+1].x - pv.x
            dh = pv.h - p[pv.index+1][1]
            texCoord = math.sqrt(self.polygonWidth_2*dx*dx + self.roofHeight_2*dh*dh)
        else: # slope is None
            texCoord = self.polygonWidth * (pv.x - slots[pv.index].x)
        return texCoord
    
    def onNewSlotVertex(self, slotIndex, vertIndex, y):
        """
        The override of the parent class method
        """
        self.texCoords.append((
            True,
            slotIndex,
            y
        ))
        # update <self.minY> and <self.maxY> if necessary
        if y < self.minY:
            self.minY = y
        elif y > self.maxY:
            self.maxY = y
    
    def onRoofForSlotCompleted(self, slotIndex):
        """
        The override of the parent class method
        """
        roofFaceToSlot = self.roofFaceToSlot
        for _ in range(len(roofFaceToSlot), len(self.roofIndices)):
            roofFaceToSlot.append(slotIndex)