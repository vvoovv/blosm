from .roof import Roof
from item.facade import Facade
from item.roof_flat import RoofFlat as ItemRoofFlat
from .geometry.rectangle import RectangleFRA
from mathutils import Vector

from util import zAxis


class RoofFlat(Roof):
    
    # default height of the top
    topHeight = 1.
    
    def __init__(self, data, itemStore, itemFactory, facadeRender, roofRenderer):
        super().__init__(data, itemStore, itemFactory)
        self.facadeRenderer = facadeRender
        self.roofRenderer = roofRenderer
        self.rectangleGeometry = RectangleFRA()
        self.hasRoofLevels = False
        self.extrudeTillRoof = False
    
    def render(self, footprint, roofItem):
        self.extrude(footprint, roofItem)
        self.facadeRenderer.render(footprint, self.data)
        self.roofRenderer.render(roofItem)
    
    def validate(self, footprint):
        """
        Additional validation
        """
        if footprint.noWalls:
            # that case can't happen for buildings with the flat roof
            footprint.valid = False
    
    def getRoofFirstVertIndex(self, footprint):
        return len(footprint.building.verts) + footprint.polygon.n
    
    def getRoofItem(self, footprint):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base;
        # since it depends on the total number of building vertices, we calculated it before any operation
        # that creates building geometry
        return ItemRoofFlat.getItem(
            self.itemFactory,
            footprint,
            self.getRoofFirstVertIndex(footprint)
        )
    
    def extrude(self, footprint, roofItem):
        building = footprint.building
        facades = footprint.facades
        verts = building.verts
        indexOffset = len(verts)
        polygon = footprint.polygon
        numVerts = polygon.n
        
        # create vertices
        z = footprint.roofVerticalPosition if self.extrudeTillRoof else footprint.height
        # verts for the lower cap
        verts.extend(v for v in polygon.verts)
        # verts for the upper cap
        verts.extend(Vector((v.x, v.y, z)) for v in polygon.verts)
        
        # the starting side
        _in = indexOffset+numVerts
        facades.append(Facade.getItem(
            self,
            footprint,
            (_in-1, indexOffset, _in, _in+numVerts-1),
            0 # edge index
        ))
        # the rest of the sides
        facades.extend(
            Facade.getItem(
                self,
                footprint,
                (indexOffset+i-1, indexOffset+i, _in+i, _in+i-1),
                i # edge index
            ) for i in range(1, numVerts)
        )
    
    def initFacadeItem(self, item):
        verts = item.building.verts
        indices = item.indices
        geometry = self.rectangleGeometry
        bottomVec = verts[indices[1]] - verts[indices[0]]
        width = bottomVec.length
        height = item.footprint.wallHeight
        
        item.width = width
        item.normal = bottomVec.cross(zAxis)/width
        item.geometry = geometry
        # assign uv-coordinates (i.e. surface coordinates on the facade plane)
        item.uvs = geometry.getUvs(width, height)
    
    def calculateRoofHeight(self, footprint):
        h = footprint.getStyleBlockAttr("topHeight")
        if h is None:
            h = footprint.getStyleBlockAttr("roofHeight")
            if h is None:
                # default height of the top
                h = self.topHeight
        footprint.levelHeights.topHeight = h
        footprint.roofHeight = 0.
        return 0.


class RoofLeveled(RoofFlat):
    """
    The base class for volume generators that generate a roof with all it roof sides starting from
    the same height.
    It is the base class for <RoofGeneratrix> and <RoofHipped>.
    """
    
    def validate(self, footprint):
        """
        Additional validation
        """
        return

    def calculateRoofHeight(self, footprint):
        h = footprint.getStyleBlockAttr("roofHeight")
        if h is None:
            # default height of the roof
            h = self.height
        footprint.roofHeight = h
        # no roof levels for now, that will be changed later
        footprint.roofLevelsHeight = 0.
        return h
    
    def getRoofFirstVertIndex(self, footprint):
        return len(footprint.building.verts) if footprint.noWalls else super().getRoofFirstVertIndex(footprint)
    
    def extrude(self, footprint, roofItem):
        if footprint.noWalls:
            z = footprint.roofVerticalPosition
            # the basement of the roof
            footprint.building.verts.extend(Vector((v.x, v.y, z)) for v in footprint.polygon.verts)
            return
        super().extrude(footprint, roofItem)