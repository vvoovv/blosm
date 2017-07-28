from . import RoofRealistic
from building.roof.profile import RoofProfile


class RoofProfileRealistic(RoofRealistic, RoofProfile):
    
    def __init__(self, data):
        super().__init__(data);
        self.texCoords = []
        # mapping between the indices
        self.indicesMap = {}
    
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

    def renderRoof(self):
        if self.mrr:
            r = self.r
            bm = r.bm
            verts = self.verts
            polygon = self.polygon
            uvLayer = bm.loops.layers.uv[0]
            texCoords = self.texCoords
            # create BMesh faces for the building roof
            for indices in self.roofIndices:
                f = bm.faces.new(verts[i] for i in indices)
                for i,roofIndex in enumerate(indices):
                    if roofIndex < polygon.indexOffset:
                        texCoords = self.texCoords[ self.indicesMap[roofIndex] ]
                    else:
                        texCoords = self.texCoords[polygon.n + roofIndex - polygon.indexOffset]
                    f.loops[i][uvLayer].uv = texCoords
                self.mrr.render(f)
        else:
            super().renderRoof()

    def getProfiledVert(self, i, roofMinHeight, noWalls):
        """
        The override of the parent class method
        """
        pv = super().getProfiledVert(i, roofMinHeight, noWalls)
        proj = self.projections
        texCoords = (
            proj[i] - proj[self.minProjIndex],
            pv.y
        )
        if pv.vertIndex < self.polygon.indexOffset:
            self.texCoords[i] = texCoords
        else:
            self.texCoords.append(texCoords)
        return pv
    
    def onNewSlotVertex(self, slotIndex, vertIndex, y):
        """
        The override of the parent class method
        """
        self.texCoords.append((
            self.slots[slotIndex].xReal,
            y
        ))