from util import zero

from . import Geometry


class Triangle(Geometry):
    
    def init(self, geometryTrapezoidH, geometryTrapezoidL, geometryTrapezoidR, geometryPolygon):
        # a classical trapezoid with the sides parallel to the horizontal axis
        self.geometryTrapezoidH = geometryTrapezoidH
        self.geometryTrapezoidL = geometryTrapezoidL
        self.geometryTrapezoidR = geometryTrapezoidR
        self.geometryPolygon = geometryPolygon
    
    def initRenderStateForLevels(self, rs, parentItem):
        super().initRenderStateForLevels(rs, parentItem)
        rs.uvBL, rs.uvBR = parentItem.uvs[0], parentItem.uvs[1]
        # <rs.startIndexL> will be used as the index of the top vertex of the triangle
        rs.startIndexL = -1
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            ( (itemUvs[2][0] - itemUvs[0][0]) / (itemUvs[1][0] - itemUvs[0][0]) * u, v)
        )
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        parentIndices = parentItem.indices
        parentUvs = parentItem.uvs
        startIndexL = rs.startIndexL
        
        verts = parentItem.building.renderInfo.verts
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        texVt = rs.texVb + height
        # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
        # an item with the trapezoid geometry to be created
        indexTL = len(verts)
        kL = (parentUvs[startIndexL][1] - texVt) / (texVt - rs.uvBL[1])
        verts.append( ( kL*verts[rs.indexBL] + verts[parentIndices[startIndexL]])/(1.+kL) )
        
        indexTR = indexTL + 1
        kR = (parentUvs[startIndexL][1] - texVt) / (texVt - rs.uvBR[1])
        verts.append( ( kR*verts[rs.indexBR] + verts[parentIndices[startIndexL]])/(1.+kR) )
        
        indices = (rs.indexBL, rs.indexBR, indexTR, indexTL)
        uvs = (
            rs.uvBL,
            rs.uvBR,
            ( (kR*rs.uvBR[0]+parentUvs[startIndexL][0])/(1.+kR), texVt ),
            ( (kL*rs.uvBL[0]+parentUvs[startIndexL][0])/(1.+kL), texVt )
        )
        
        item = levelGroup.item
        if item:
            # Set the geometry for the <levelGroup.item>;
            # division of a triangle can only generate a classical trapezoid at the bottom and
            # a triangle at the top
            item.geometry = self.geometryTrapezoidH
        
        if item and item.markup:
            item.indices = indices
            item.uvs = uvs
            levelRenderer.renderDivs(item, levelGroup)
        else:
            levelRenderer.renderLevelGroup(
                item or parentItem,
                levelGroup,
                indices,
                uvs
            )
        rs.indexBL = indexTL
        rs.indexBR = indexTR
        rs.texVb = texVt
        rs.uvBL, rs.uvBR = uvs[3], uvs[2]
    
    def renderLastLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        item = levelGroup.item
        if levelGroup.item:
            levelGroup.item.geometry = self
            
        if item and item.markup:
            item.indices = (rs.indexBL, rs.indexBR, parentItem.indices[rs.startIndexL])
            item.uvs = ( rs.uvBL, rs.uvBR, parentItem.uvs[rs.startIndexL] )
            levelRenderer.renderDivs(item, levelGroup)
        else:
            levelRenderer.renderLevelGroup(
                item or parentItem,
                levelGroup,
                (rs.indexBL, rs.indexBR, parentItem.indices[rs.startIndexL]),
                (rs.uvBL, rs.uvBR, parentItem.uvs[rs.startIndexL])
            )
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        parentRenderer.renderCladding(
            parentItem,
            parentRenderer.r.createFace(
                parentItem.footprint,
                (rs.indexBL, rs.indexBR, parentItem.indices[rs.startIndexL])
            ),
            ( rs.uvBL, rs.uvBR, parentItem.uvs[rs.startIndexL] )
        )
    
    def offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset):
        raL = parentUvs[0][0] == parentUvs[2][0]
        # only one of <raL> or <raR> can be <True>
        raR = False if raL else (parentUvs[1][0] == parentUvs[2][0])
        
        verts = item.building.renderInfo.verts
        
        # the new vertex at the bottom
        indexB = len(verts)
        Geometry._appendVertAtBottom(verts, parentIndices, parentUvs, offset)
        # add offset
        offsetU = parentUvs[0][0] + offset
        uvB = (offsetU, parentUvs[0][1])
        
        # initialize the variables to be used below in the code
        indexT = 0
        uvT = indicesOffset = uvsOffset = indicesGeometry = uvsGeometry = None
        
        if raL:
            # we don't need to change <item.geometry>, it remains to be <Triangle>
            indexT, uvT = Geometry._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
            
            # the offset area has the geometry of <TrapezoidRV>
            indicesOffset = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
            uvsOffset = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
            
            indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
            uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
        elif raR:
            # change <item.geometry> to <TrapezoidRV>
            item.geometry = self.geometryTrapezoidL
            indexT, uvT = Geometry._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
            
            # the offset area has the geometry of <Triangle>
            indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
            uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
            
            indicesGeometry = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
            uvsGeometry = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
        else:
            #
            # the general case
            # 
            if offsetU < parentUvs[2][0] - zero:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = Geometry._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Triangle._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Triangle._getUvsPolygonAtRight(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[2][0] + zero:
                # we don't need to change <item.geometry>, it remains to be <Triangle>
                # use existing vertex
                indexT = parentIndices[2]
                uvT = parentUvs[2]
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
            else:
                # we don't need to change <item.geometry>, it remains to be <Triangle>
                indexT, uvT = Geometry._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Triangle._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Triangle._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
        
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)
    
    def offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset):
        raL = parentUvs[0][0] == parentUvs[2][0]
        # only one of <raL> or <raR> can be <True>
        raR = False if raL else (parentUvs[1][0] == parentUvs[2][0])
        
        verts = item.building.renderInfo.verts
        
        # convert <offset> to the distance from the leftmost vertex
        offset = parentUvs[1][0] - parentUvs[0][0] - offset
        
        # the new vertex at the bottom
        indexB = len(verts)
        Geometry._appendVertAtBottom(verts, parentIndices, parentUvs, offset)
        # add offset
        offsetU = parentUvs[0][0] + offset
        uvB = (offsetU, parentUvs[0][1])
        
        # initialize the variables to be used below in the code
        indexT = 0
        uvT = indicesOffset = uvsOffset = indicesGeometry = uvsGeometry = None
        
        if raL:
            # change <item.geometry> to <TrapezoidRV>
            item.geometry = self.geometryTrapezoidR
            indexT, uvT = Geometry._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
            
            # the offset area has the geometry of <Triangle>
            indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
            uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
            
            indicesGeometry = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
            uvsGeometry = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
        elif raR:
            # we don't need to change <item.geometry>, it remains to be <Triangle>
            indexT, uvT = Geometry._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
            
            # the offset area has the geometry of <TrapezoidRV>
            indicesOffset = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
            uvsOffset = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
            
            indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
            uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
        else:
            #
            # the general case
            # 
            if offsetU < parentUvs[2][0] - zero:
                # we don't need to change <item.geometry>, it remains to be <Triangle>
                indexT, uvT = Geometry._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Triangle._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsOffset = Triangle._getUvsPolygonAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[2][0] + zero:
                # we don't need to change <item.geometry>, it remains to be <Triangle>
                # use existing vertex
                indexT = parentIndices[2]
                uvT = parentUvs[2]
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = Geometry._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Triangle._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Triangle._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
        
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)
    
    @staticmethod
    def _getIndicesPolygonAtLeft(indexB, indexT, parentIndices):
        return (parentIndices[0], indexB, indexT, parentIndices[2])
    
    @staticmethod
    def _getUvsPolygonAtLeft(uvB, uvT, parentUvs):
        return (parentUvs[0], uvB, uvT, parentUvs[2])
    
    @staticmethod
    def _getIndicesPolygonAtRight(indexB, indexT, parentIndices):
        return (indexB, parentIndices[1], parentIndices[2], indexT)

    @staticmethod
    def _getUvsPolygonAtRight(uvB, uvT, parentUvs):
        return (uvB, parentUvs[1], parentUvs[2], uvT)