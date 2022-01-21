from .roof import Roof
from item.facade import Facade
from item.roof_flat import RoofFlat as ItemRoofFlat
from .geometry.rectangle import RectangleFRA
from mathutils import Vector

from util import zAxis


class RoofFlat(Roof):
    
    # default height of the top
    topHeight = 1.
    
    def __init__(self, roofRendererId, data, volumeAction, itemRenderers):
        super().__init__(roofRendererId, data, volumeAction, itemRenderers)
        self.rectangleGeometry = RectangleFRA()
        self.hasRoofLevels = False
        self.extrudeTillRoof = False
    
    def render(self, footprint, roofItem):
        self.extrude(footprint, roofItem)
        self.facadeRenderer.render(footprint)
        self.roofRenderer.render(roofItem)
    
    def validate(self, footprint):
        """
        Additional validation
        """
        if footprint.noWalls:
            # that case can't happen for buildings with the flat roof
            footprint.valid = False
    
    def getRoofFirstVertIndex(self, footprint):
        return len(footprint.building.renderInfo.verts) + footprint.polygon.numEdges
    
    def getRoofItem(self, footprint):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base;
        # since it depends on the total number of building vertices, we calculated it before any operation
        # that creates building geometry
        return ItemRoofFlat(
            footprint,
            self.getRoofFirstVertIndex(footprint)
        )
    
    def extrude(self, footprint, roofItem):
        building = footprint.building
        verts = building.renderInfo.verts
        indexOffset = len(verts)
        polygon = footprint.polygon
        numVerts = polygon.numEdges
        
        # create vertices
        
        # verts for the lower cap
        z = footprint.minHeight
        verts.extend(Vector((v[0], v[1], z)) for v in polygon.verts)
        # verts for the upper cap
        z = footprint.roofVerticalPosition if self.extrudeTillRoof else footprint.height
        verts.extend(Vector((v[0], v[1], z)) for v in polygon.verts)
        
        vectors = polygon.getVectors()
        
        _in = indexOffset+numVerts
        # the first <numVerts-1> edges
        footprint.facades.extend(
            Facade(
                footprint,
                (indexOffset+i, indexOffset+i+1, _in+i+1, _in+i),
                vector, # edge index
                self
            ) for i,vector in zip(range(numVerts-1), vectors)
        )
        
        # the closing edge
        footprint.facades.append(
            Facade(
                footprint,
                (_in-1, indexOffset, _in, _in+numVerts-1),
                next(vectors),
                self
            )
        )
    
    def initFacadeItem(self, item):
        verts = item.building.renderInfo.verts
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
        return len(footprint.building.renderInfo.verts) if footprint.noWalls else super().getRoofFirstVertIndex(footprint)
    
    def extrude(self, footprint, roofItem):
        if footprint.noWalls:
            z = footprint.roofVerticalPosition
            # the basement of the roof
            footprint.building.renderInfo.verts.extend(Vector((v.x, v.y, z)) for v in footprint.polygon.verts)
            return
        super().extrude(footprint, roofItem)