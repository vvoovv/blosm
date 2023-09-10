from util import zero, zAxis

from . import Geometry
from .rectangle import RectangleFRA
from .triangle import Triangle
from .polygon import PolygonHB


class TrapezoidH(Geometry):
    """
    A classical trapezoid with parallel sides along the horizontal axis
    """
    def init(self, geometryTriangle, geometryRectangle, geometryPolygon):
        self.geometryTriangle = geometryTriangle
        self.geometryRectangle = geometryRectangle
        self.geometryPolygon = geometryPolygon
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        
        # Calculate the width of <item> as the difference between the largest U-coordinate and
        # the smallest one
        factorU = u/(itemUvs[1][0] - itemUvs[0][0])
        
        return (
            (0., 0.,),
            (u, 0.),
            ( (itemUvs[2][0]-itemUvs[0][0])*factorU, v),
            ( (itemUvs[3][0]-itemUvs[0][0])*factorU, v)
        )
    
    def offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset):
        raL = parentUvs[0][0] == parentUvs[-1][0]
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
            if offsetU <= parentUvs[2][0] + zero:
                if offsetU >= parentUvs[2][0] - zero:
                    # change <item.geometry> to <Triangle>
                    item.geometry = self.geometryTriangle
                    # use existing vertex
                    indexT = parentIndices[2]
                    uvT = parentUvs[2]
                    
                    indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                    uvsGeometry = Geometry._getUvsTriangleAtRight(self, uvB, uvT, parentUvs)
                else:
                    # We don't need to change <item.geometry>, it remains to be <TrapezoidH>.
                    # Add a point on the rectangle.
                    indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleL(verts, parentIndices, parentUvs, offset)
                    
                    indicesGeometry = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                    uvsGeometry = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
                # the offset area has the geometry of <Rectangle>
                indicesOffset = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                indexT, uvT = TrapezoidH._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Geometry._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
        elif raR:
            if offsetU >= parentUvs[3][0] - zero:
                if offsetU <= parentUvs[3][0] + zero:
                    # use existing vertex
                    indexT = parentIndices[3]
                    uvT = parentUvs[3]
                    
                    # the offset area has the geometry of <Triangle>
                    indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                    uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                else:
                    indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleR(verts, parentIndices, parentUvs, offsetU)
                    
                    # the offset area has the geometry of <TrapezoidH>
                    indicesOffset = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                    uvsOffset = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
                
                # change <item.geometry> to <Rectangle>
                item.geometry = self.geometryRectangle
                indicesGeometry = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = TrapezoidH._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsPolygonAtRight(uvB, uvT, parentUvs)
        else:
            #
            # the general case
            # 
            if offsetU < parentUvs[3][0] - zero:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = TrapezoidH._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsPolygonAtRight(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[3][0] + zero:
                # we don't need to change <item.geometry>, it remains to be <TrapezoidH>
                # use existing vertex
                indexT = parentIndices[3]
                uvT = parentUvs[3]
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
            elif offsetU < parentUvs[3][0] - zero:
                # we don't need to change <item.geometry>, it remains to be <TrapezoidH>
                
                indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleR(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <TrapezoidH>
                indicesOffset = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[3][0] + zero:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                # use existing vertex
                indexT = parentIndices[2]
                uvT = parentUvs[2]
                
                # the offset area has the geometry of <TrapezoidH>
                indicesOffset = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                indexT, uvT = TrapezoidH._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Geometry._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)
    
    def offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset):
        raL = parentUvs[0][0] == parentUvs[-1][0]
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
            if offsetU <= parentUvs[2][0] + zero:
                if offsetU >= parentUvs[2][0] - zero:
                    # use existing vertex
                    indexT = parentIndices[2]
                    uvT = parentUvs[2]
                    
                    indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                    uvsOffset = Geometry._getUvsTriangleAtRight(self, uvB, uvT, parentUvs)
                else:
                    indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleL(verts, parentIndices, parentUvs, offset)
                    
                    indicesOffset = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                    uvsOffset = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
                # change <item.geometry> to <Rectangle>
                item.geometry = self.geometryRectangle
                indicesGeometry = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = TrapezoidH._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
        elif raR:
            if offsetU >= parentUvs[3][0] - zero:
                if offsetU <= parentUvs[3][0] + zero:
                    # change <item.geometry> to <Triangle>
                    item.geometry = self.geometryTriangle
                    # use existing vertex
                    indexT = parentIndices[3]
                    uvT = parentUvs[3]
                    
                    indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                    uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
                else:
                    # we don't need to change <item.geometry>, it remains to be <TrapezoidH>
                    
                    indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleR(verts, parentIndices, parentUvs, offsetU)
                    
                    # the offset area has the geometry of <TrapezoidH>
                    indicesGeometry = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                    uvsGeometry = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
                
                # the offset area has the geometry of <Rectangle>
                indicesOffset = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                indexT, uvT = TrapezoidH._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Geometry._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsPolygonAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
        else:
            #
            # the general case
            # 
            if offsetU < parentUvs[3][0] - zero:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                indexT, uvT = TrapezoidH._getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <PolygonHB>
                indicesOffset = Geometry._getIndicesPolygonAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsPolygonAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[3][0] + zero:
                # change <item.geometry> to <Triangle>
                item.geometry = self.geometryTriangle
                # use existing vertex
                indexT = parentIndices[3]
                uvT = parentUvs[3]
                
                # the offset area has the geometry of <TrapezoidH>
                indicesOffset = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTriangleAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTriangleAtLeft(uvB, uvT, parentUvs)
            elif offsetU < parentUvs[3][0] - zero:
                # we don't need to change <item.geometry>, it remains to be <TrapezoidH>
                
                indexT, uvT = TrapezoidH._getIndexAndUvAtRectangleR(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <TrapezoidH>
                indicesOffset = Geometry._getIndicesTrapezoidAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTrapezoidAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
            elif offsetU <= parentUvs[3][0] + zero:
                # we don't need to change <item.geometry>, it remains to be <TrapezoidH>
                
                # use existing vertex
                indexT = parentIndices[2]
                uvT = parentUvs[2]
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsTrapezoidAtLeft(uvB, uvT, parentUvs)
            else:
                # change <item.geometry> to <PolygonHB>
                item.geometry = self.geometryPolygon
                indexT, uvT = TrapezoidH._getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU)
                
                # the offset area has the geometry of <Triangle>
                indicesOffset = Geometry._getIndicesTriangleAtRight(indexB, indexT, parentIndices)
                uvsOffset = Geometry._getUvsTriangleAtRight(uvB, uvT, parentUvs)
                
                indicesGeometry = Geometry._getIndicesPolygonAtLeft(indexB, indexT, parentIndices)
                uvsGeometry = Geometry._getUvsPolygonAtLeft(uvB, uvT, parentUvs)
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)
            
    @staticmethod
    def _getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU):
        """
        A helper function
        """
        indexT = len(verts)
        k = (offsetU-parentUvs[0][0]) / (parentUvs[3][0]-parentUvs[0][0])
        verts.append(
            verts[parentIndices[0]] + k * (verts[parentIndices[3]] - verts[parentIndices[0]])
        )
        return\
            indexT,\
            (#uvT
                offsetU,
                parentUvs[0][1] + k * (parentUvs[3][1] - parentUvs[0][1])
            )
    
    @staticmethod
    def _getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU):
        """
        A helper function
        """
        indexT = len(verts)
        k = (offsetU-parentUvs[2][0]) / (parentUvs[1][0]-parentUvs[2][0])
        verts.append(
            verts[parentIndices[2]] + k * (verts[parentIndices[1]] - verts[parentIndices[2]])
        )
        return\
            indexT,\
            (
                offsetU,
                parentUvs[2][1] + k * (parentUvs[1][1] - parentUvs[2][1])
            )
    
    @staticmethod
    def _getIndexAndUvAtRectangleL(verts, parentIndices, parentUvs, offset):
        indexT = len(verts)
        verts.append(
            verts[parentIndices[3]] + \
                offset/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        )
        return indexT, (offset, parentUvs[3][1])

    @staticmethod
    def _getIndexAndUvAtRectangleR(verts, parentIndices, parentUvs, offsetU):
        """
        A helper function
        """
        indexT = len(verts)
        verts.append(
            verts[parentIndices[3]] +\
                (offsetU-parentIndices[3])/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        )
        return indexT, (offsetU, parentUvs[3][1])


class TrapezoidRV(Geometry):
    """
    A right-angled trapezoid with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """

    def __init__(self, leftIsLower, geometryTrapezoidChained):
        self.leftIsLower = leftIsLower
        self.geometryRectangle = geometryTrapezoidChained.geometryRectangle
        self.geometryTriangle = geometryTrapezoidChained.geometryTriangle
        self.geometryTrapezoidChained = geometryTrapezoidChained
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        
        # trapezoid height on the left and right
        heightL = itemUvs[-1][1] - itemUvs[0][1]
        heightR = itemUvs[ 2][1] - itemUvs[1][1]
        
        return (
            (0., 0.,),
            (u, 0.),
            (u, v if self.leftIsLower else heightR*v/heightL),
            (0., heightL*v/heightR if self.leftIsLower else v)
        )

    def getClassUvs(self, texUl, texVb, texUr, texVt, uvs):
        # Is the V-coordiante of uvs[2] is larger than the one of uvs[2]
        isV2LargerV3 = uvs[2][1] > uvs[3][1]
        return (
            (texUl, texVb),
            (texUr, texVb),
            (texUr, texVt if isV2LargerV3 else texVb + (texVt-texVb)*(uvs[2][1] - uvs[1][1])/(uvs[3][1] - uvs[0][1])),
            (texUl, texVb + (texVt-texVb)*(uvs[3][1] - uvs[0][1])/(uvs[2][1] - uvs[1][1]) if isV2LargerV3 else texVt)
        )
    
    def renderDivs(self,
            itemRenderer, building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            rs
        ):
        # <startIndex> is not used by the <TrapezoidRV> geometry
        verts = building.verts
        indexLB = rs.indexLB
        indexLT = rs.indexLT
        texUl = rs.texUl
        texVlt = rs.texVlt
        # <texVb> is the V-coordinate for the bottom vertices of the trapezoid items
        # to be created out of <item>
        texVb = item.uvs[0][1]
        v1 = verts[indexLB]
        v2 = verts[indexLT]
        # <factor> is used in the calculations below, it is actually a tangens of the trapezoid angle
        factor = (item.uvs[2][1] - item.uvs[3][1])/item.width
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            # Set the geometry for the <_item>; division of a trapezoid can only generate trapezoids
            _item.geometry = self
            # <indexRB> and <indexRT> are indices of the bottom and top vertices
            # on the right side of an item with rectangular geometry to be created
            # Additional vertices can be created inside <_item.getItemRenderer(itemRenderer.itemRenderers).render(..)>,
            # that's why we use <len(building.verts)>
            indexRB = len(building.verts)
            indexRT = indexRB + 1
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            v2 = v2 + incrementVector
            v2[2] += _item.width * factor
            verts.append(v2)
            texUr = texUl + _item.width
            texVrt = texVlt + _item.width * factor
            _item.getItemRenderer(itemRenderer.itemRenderers).render(
                _item,
                (indexLB, indexRB, indexRT, indexLT),
                ( (texUl, texVb), (texUr, texVb), (texUr, texVrt), (texUl, texVlt) )
            )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
            texVlt = texVrt
        
        rs.indexLB = indexLB
        rs.indexLT = indexLT
        rs.texUl = texUl
        rs.texVlt = texVlt

    def renderLastDiv(self, itemRenderer, parentItem, lastItem, rs):
        # <startIndex> is not used by the <TrapezoidRV> geometry
        parentIndices = parentItem.indices
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <parentItem>
        texUr = parentItem.uvs[1][0]
        # <texVb> is the V-coordinate for bottom vertices of the trapezoid item
        # to be created out of <parentItem>
        texVb = parentItem.uvs[0][1]
        # Set the geometry for the <lastItem>; division of a trapezoid can only generate trapezoids
        lastItem.geometry = self
        lastItem.getItemRenderer(itemRenderer.itemRenderers).render(
            lastItem,
            (rs.indexLB, parentIndices[1], parentIndices[2], rs.indexLT),
            ( (rs.texUl, texVb), (texUr, texVb), (texUr, parentItem.uvs[2][1]), (rs.texUl, rs.texVlt) )
        )

    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        if rs.remainingGeometry:
            rs.remainingGeometry.renderLevelGroup(parentItem, levelGroup, levelRenderer, rs)
            return
        elif levelGroup.index2 < parentItem.footprint.numLevels-1:
            # Comparison between the integers is faster than between floats.
            # Treat it like a rectangle right away
            self.geometryRectangle.renderLevelGroup(
                parentItem, levelGroup, levelRenderer, rs
            )
            return
        
        verts = parentItem.building.renderInfo.verts
        parentIndices = parentItem.indices
        parentUvs = parentItem.uvs
        # initialize the variables <indexTL> and <indexTR>
        indexTL = indexTR = 0
        # initialize a UV coordinate to be used later
        _uv = None
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        texVt = rs.texVb + height
        
        # Note that even if <levelGroup.index2 == parentItem.numLevels>, the height of the last level
        # with the index equal to <parentItem.numLevels> can eventually exceed the height of the vertices
        # with the indices <-1> and <2> when <lastLevelOffsetFactor> is greater than zero.
        
        if self.leftIsLower:
            # create the top right vertex
            indexTR = len(verts)
            verts.append(verts[rs.indexBR] + height*zAxis)
            
            if texVt <= parentUvs[3][1] + zero:
                # the case of rectangle
                if texVt >= parentUvs[3][1] - zero:
                    # use existing vertex
                    indexTL = parentIndices[3]
                    # the remaining geometry is a triangle
                    rs.remainingGeometry = self.geometryTriangle
                    # <rs.startIndexL> will be used as the index of the top vertex of the triangle
                    rs.startIndexL = 2
                    rs.uvBL, rs.uvBR = parentUvs[3], (parentUvs[1][0], texVt)
                else:
                    indexTL = len(verts)
                    verts.append(verts[rs.indexBL] + height*zAxis)
                # render <levelGroup> in a rectangular geometry
                self.geometryRectangle._renderLevelGroupRectangle(
                    parentItem, levelGroup, levelRenderer, rs, indexTL, indexTR, texVt
                )
                return
            else:
                indexTL = len(verts)
                # factor
                k = (parentUvs[2][1] - texVt) / (texVt - parentUvs[3][1])
                verts.append(
                    (k*verts[parentIndices[3]] + verts[parentIndices[2]])/(1.+k)
                )
                _uv = (
                    (k*parentUvs[3][0] + parentUvs[2][0])/(1.+k),
                    (k*parentUvs[3][1] + parentUvs[2][1])/(1.+k)
                )
                # the remaining geometry is a triangle
                rs.remainingGeometry = self.geometryTriangle
                # <rs.startIndexL> will be used as the index of the top vertex of the triangle
                rs.startIndexL = 2
                rs.uvBL, rs.uvBR = _uv, (parentUvs[1][0], texVt)
        else:
            # create the top left vertex
            indexTL = len(verts)
            verts.append(verts[rs.indexBL] + height*zAxis)
            if texVt <= parentUvs[2][1] + zero:
                # the case of rectangle
                if texVt >= parentUvs[2][1] - zero:
                    # use existing vertex
                    indexTR = parentIndices[2]
                    # the remaining geometry is a triangle
                    rs.remainingGeometry = self.geometryTriangle
                    # <rs.startIndexL> will be used as the index of the top vertex of the triangle
                    rs.startIndexL = 3
                    rs.uvBL, rs.uvBR = (parentUvs[0][0], texVt), parentUvs[2]
                else:
                    indexTR = len(verts)
                    verts.append(verts[rs.indexBR] + height*zAxis)
                # render <levelGroup> in a rectangular geometry
                self.geometryRectangle._renderLevelGroupRectangle(
                    parentItem, levelGroup, levelRenderer, rs, indexTL, indexTR, texVt
                )
                return
            else:
                indexTR = len(verts)
                # factor
                k = (parentUvs[3][1] - texVt) / (texVt - parentUvs[2][1])
                verts.append(
                    (k*verts[parentIndices[2]] + verts[parentIndices[3]])/(1.+k)
                )
                _uv = (
                    (k*parentUvs[2][0] + parentUvs[3][0])/(1.+k),
                    (k*parentUvs[2][1] + parentUvs[3][1])/(1.+k)
                )
                # the remaining geometry is a triangle
                rs.remainingGeometry = self.geometryTriangle
                # <rs.startIndexL> will be used as the index of the top vertex of the triangle
                rs.startIndexL = 3
                rs.uvBL, rs.uvBR = (parentUvs[0][0], texVt), _uv
        
        item = levelGroup.item
        if item:
            # Set the geometry for the <levelGroup.item>. Since the geometry
            # has 5 points with the vertical sides, it is supposed to be <self.geometryTrapezoidChained>
            item.geometry = self.geometryTrapezoidChained
        
        indices = (rs.indexBL, rs.indexBR, indexTR, indexTL, parentIndices[3])\
            if self.leftIsLower else\
            (rs.indexBL, rs.indexBR, parentIndices[2], indexTR, indexTL)
        
        uvs = ((parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), (parentUvs[1][0], texVt), _uv, parentUvs[3])\
            if self.leftIsLower else\
            ((parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), parentUvs[2], _uv, (parentUvs[0][0], texVt))
        
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
    
    def renderLastLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        if rs.remainingGeometry:
            rs.remainingGeometry.renderLastLevelGroup(parentItem, levelGroup, levelRenderer, rs)
        else:
            parentIndices = parentItem.indices
            texVt = parentItem.uvs[2][1]
            # <texUl> and <texUr> are the left and right U-coordinates for the rectangular items
            # to be created out of <parentItem>
            texUl = parentItem.uvs[0][0]
            texUr = parentItem.uvs[1][0]
            
            item = levelGroup.item
            if levelGroup.item:
                # Set the geometry for the <group.item>;
                # division of a rectangle can only generate rectangles
                levelGroup.item.geometry = self
                
            if item and item.markup:
                item.indices = (rs.indexBL, rs.indexBR, parentIndices[2], parentIndices[3])
                item.uvs = ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
                levelRenderer.renderDivs(item, levelGroup)
            else:
                levelRenderer.renderLevelGroup(
                    item or parentItem,
                    levelGroup,
                    (rs.indexBL, rs.indexBR, parentIndices[2], parentIndices[3]),
                    ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
                )
    
    def subtract(self, facade, _facade):
        return
    
    def join(self, facade, _facade):
        return
    
    def getMaxV(self, uvs):
        return uvs[2][1] if uvs[2][1] > uvs[3][1] else uvs[3][1]
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        if rs.remainingGeometry:
            rs.remainingGeometry.renderCladdingAtTop(parentItem, parentRenderer)
        else:
            # <texUl> and <texUr> are the left and right U-coordinates for the rectangular items
            # to be created out of <parentItem>
            texUl = parentItem.uvs[0][0]
            texUr = parentItem.uvs[1][0]
            self._renderCladding(
                parentItem,
                parentRenderer,
                (rs.indexBL, rs.indexBR, parentItem.indices[2], parentItem.indices[3]),
                ( (texUl, rs.texVb), (texUr, rs.texVb), parentItem.uvs[2], parentItem.uvs[3] )
            )


class TrapezoidChainedRV(Geometry):
    """
    A sequence of adjoining right-angled trapezoids with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """
    
    def __init__(self):
        trapezoidH = TrapezoidH()
        self.geometryRectangle = RectangleFRA()
        self.geometryTriangle = Triangle(trapezoidH)
        self.geometryTrapezoidL = TrapezoidRV(True, self) # Left is lower than right
        self.geometryTrapezoidR = TrapezoidRV(False, self) # Right is lower than left
        self.geometryPolygon = PolygonHB(self.geometryTriangle, self.geometryTrapezoidL, self.geometryTrapezoidR)
        trapezoidH.init(self.geometryTriangle, self.geometryRectangle, self.geometryPolygon)
    
    def initRenderStateForLevels(self, rs, parentItem):
        super().initRenderStateForLevels(rs, parentItem)
        rs.startIndexL = -1
        rs.startIndexR = 2

    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        offsetU, offsetV = itemUvs[0]
        # Calculate the height of <item> as the difference between the largest V-coordinate and
        # the smallest one
        h = max(itemUvs[i][1] for i in range(3, len(itemUvs)-1)) - offsetV
        # Calculate the width of <item> as the difference between the largest U-coordinate and
        # the smallest one
        w = itemUvs[1][0] - itemUvs[0][0]
        
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        
        factorU, factorV = u/w, v/h
        
        uvs = [
            (0., 0.,),
            (u, 0.),
            (u, (itemUvs[2][1] - offsetV)*factorV)
        ]
        uvs.extend(
            ( (itemUvs[i][0]-offsetU)*factorU, (itemUvs[i][1]-offsetV)*factorV )\
                for i in range(3, len(itemUvs)-1)
        )
        uvs.append(
            (0., (itemUvs[-1][1] - offsetV)*factorV)
        )
        
        return uvs
    
    def initRenderStateForDivs(self, rs, item):
        super().initRenderStateForDivs(rs, item)
        rs.startIndex = len(item.indices) - 1
    
    def renderDivs(self,
            itemRenderer, item, levelGroup, unitVector, markupItemIndex1, markupItemIndex2, step,
            rs
        ):
        # <startIndex> is used for optimization
        verts = item.building.verts
        indices = item.indices
        uvs = item.uvs
        indexLB = rs.indexLB
        indexLT = rs.indexLT
        texUl = rs.texUl
        texVlt = rs.texVlt
        startIndex = rs.startIndex
        # <texVb> is the V-coordinate for the bottom vertices of the trapezoid items
        # to be created out of <item>
        texVb = item.uvs[0][1]
        v1 = verts[indexLB]
        v2 = verts[indexLT]
        stopIndex = startIndex - 1
        stopIndexPlus1 = startIndex
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            chainedTrapezoid = False
            _item = item.markup[_i]
            # <indexRB> and <indexRT> are indices of the bottom and top vertices
            # on the right side of an item with rectangular geometry to be created
            # Additional vertices can be created inside <_item.getItemRenderer(itemRenderer.itemRenderers).render(..)>,
            # that's why we use <len(building.verts)>
            indexRB = len(item.building.verts)
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            # Find vertex index of the chained trapezoid represented by <item>
            # which U-coordinate is between <texUl> and <texUr>
            texUr = texUl + _item.width
            while True:
                if texUr <= uvs[stopIndex][0]+zero:
                    if abs(uvs[stopIndex][0]-texUr) < zero:
                        indexRT = indices[stopIndex]
                        # It means the vertex of <item> with the index <stopIndex> lies
                        # on the <_item>'s right border
                        v2 = verts[indexRT]
                        texUr, texVrt = uvs[stopIndex]
                        if chainedTrapezoid:
                            _indices = (indexLB, indexRB)
                            _uvs = ( (texUl, texVb), (texUr, texVb) )
                        startIndex = stopIndex - 1
                    else:
                        indexRT = indexRB + 1
                        v2 = v2 + incrementVector
                        verticalIncrement = (texUr - uvs[stopIndexPlus1][0] if chainedTrapezoid else _item.width) *\
                            (uvs[stopIndex][1] - uvs[stopIndexPlus1][1])/(uvs[stopIndex][0] - uvs[stopIndexPlus1][0])
                        v2[2] = (verts[indices[stopIndexPlus1]][2] if chainedTrapezoid else v2[2]) + verticalIncrement
                        texVrt = (uvs[stopIndexPlus1][1] if chainedTrapezoid else texVlt) + verticalIncrement
                        verts.append(v2)
                        if chainedTrapezoid:
                            _indices = (indexLB, indexRB, indexRT)
                            _uvs = ( (texUl, texVb), (texUr, texVb), (texUr, texVrt) )
                    break
                else:
                    stopIndex -= 1
                    stopIndexPlus1 -= 1
                    if not chainedTrapezoid:
                        chainedTrapezoid = True
            
            if chainedTrapezoid:
                _item.geometry = self
                _item.getItemRenderer(itemRenderer.itemRenderers).render(
                    _item,
                    levelGroup,
                    _indices + tuple( indices[i] for i in range(stopIndexPlus1, startIndex) ) + (indexLT,),
                    _uvs + tuple( uvs[i] for i in range(stopIndexPlus1, startIndex) ) + ((texUl, texVlt),)
                )
            else:
                _item.geometry = self.geometryTrapezoid
                _item.getItemRenderer(itemRenderer.itemRenderers).render(
                    _item,
                    levelGroup,
                    (indexLB, indexRB, indexRT, indexLT),
                    ( (texUl, texVb), (texUr, texVb), (texUr, texVrt), (texUl,texVlt) )
                )
            
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
            texVlt = texVrt
            startIndex = stopIndexPlus1
        
        rs.indexLB = indexLB
        rs.indexLT = indexLT
        rs.texUl = texUl
        rs.texVlt = texVlt
        rs.startIndex = startIndex
    
    def renderLastDiv(self, itemRenderer, parenItem, lastItem, rs):
        parentIndices = parenItem.indices
        uvs = parenItem.uvs
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <parenItem>
        texUr = parenItem.uvs[1][0]
        # <texVb> is the V-coordinate for bottom vertices of the trapezoid item
        # to be created out of <parenItem>
        texVb = parenItem.uvs[0][1]
        chainedTrapezoid = rs.startIndex > 3
        if chainedTrapezoid:
            lastItem.geometry = self
            lastItem.getItemRenderer(itemRenderer.itemRenderers).render(
                lastItem,
                # indices
                (rs.indexLB, parentIndices[1], parentIndices[2]) +\
                    tuple( parentIndices[i] for i in range(3, rs.startIndex) ) +\
                    (rs.indexLT,),
                # UV-coordinates
                ( (rs.texUl, texVb), (texUr, texVb), (texUr, parenItem.uvs[2][1]) ) +\
                    tuple( uvs[i] for i in range(3, rs.startIndex) ) +\
                    ((rs.texUl, rs.texVlt),)
            )
        else:
            lastItem.geometry = self.geometryTrapezoid
            lastItem.getItemRenderer(itemRenderer.itemRenderers).render(
                lastItem,
                # indices
                (rs.indexLB, parentIndices[1], parentIndices[2], rs.indexLT),
                # UV-coordinates
                ( (rs.texUl, texVb), (texUr, texVb), (texUr, parenItem.uvs[2][1]), (rs.texUl, rs.texVlt) )
            )
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        if rs.remainingGeometry:
            rs.remainingGeometry.renderLevelGroup(parentItem, levelGroup, levelRenderer, rs)
            return
        elif levelGroup.index2 < parentItem.footprint.numLevels-1:
            # Comparison between the integers is faster than between floats.
            # Treat it like a rectangle right away
            self.geometryRectangle.renderLevelGroup(
                parentItem, levelGroup, levelRenderer, rs
            )
            return
        
        verts = parentItem.building.renderInfo.verts
        parentIndices = parentItem.indices
        parentUvs = parentItem.uvs
        startIndexL, startIndexR = rs.startIndexL, rs.startIndexR
        # initialize the variables <indexTL> and <indexTR>
        indexTL = indexTR = 0
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        texVt = rs.texVb + height
        
        # Note that even if <levelGroup.index2 == parentItem.numLevels>, the height of the last level
        # with the index equal to <parentItem.numLevels> can eventually exceed the height of the vertices
        # with the indices <-1> and <2> when <lastLevelOffsetFactor> is greater than zero.
        
        # <berasb> stands for Below or Equal Right Angle Side at the Bottom
        berasbL = False
        # <aerasb> stands for Above or Equal Right Angle Side at the Bottom
        aerasbL = True
        indicesL = uvsL = None
        # Check the condition for the left side
        if texVt <= parentUvs[-1][1] + zero:
            berasbL = True
            # initialize a UV coordinate to be used later 
            _uv = None
            if texVt >= parentUvs[-1][1] - zero:
                # use existing vertex
                indexTL = parentIndices[-1]
                _uv = parentUvs[-1]
                startIndexL = -2
            else:
                aerasbL = False
                indexTL = len(verts)
                verts.append(verts[rs.indexBL] + height*zAxis)
                _uv = (0., texVt)
            indicesL = [indexTL]
            uvsL = [_uv]
        else:
            indicesL = [parentIndices[-1]]
            uvsL = [parentUvs[-1]]
            startIndexL = -2
            while True:
                if texVt <= parentUvs[startIndexL][1] + zero:
                    if texVt >= parentUvs[startIndexL][1] - zero:
                        # use existing vertex
                        indexTL = parentIndices[startIndexL]
                        uvsL.append(parentUvs[startIndexL])
                        startIndexL -= 1
                    else:
                        indexTL = len(verts)
                        # factor
                        k = (parentUvs[startIndexL][1] - texVt) / (texVt - parentUvs[startIndexL+1][1])
                        verts.append(
                            (k*verts[parentIndices[startIndexL+1]] + verts[parentIndices[startIndexL]])/(1.+k)
                        )
                        uvsL.append((
                            (k*parentUvs[startIndexL+1][0] + parentUvs[startIndexL][0])/(1.+k),
                            (k*parentUvs[startIndexL+1][1] + parentUvs[startIndexL][1])/(1.+k)
                        ))
                    indicesL.append(indexTL)
                    break
                indicesL.append(parentIndices[startIndexL])
                uvsL.append(parentUvs[startIndexL])
                startIndexL -= 1

        # Check the condition for the left side
        # <berasb> stands for Below or Equal Right Angle Side at the Bottom
        berasbR = False
        # <aerasb> stands for Above or Equal Right Angle Side at the Bottom
        aerasbR = True
        indicesR = uvsR = None
        # the following variable is used if <rs.remainingGeometry> is set
        _uvIndex = 0
        if texVt <= parentUvs[2][1] + zero:
            # the case of the right angle
            berasbR = True
            # initialize a UV coordinate to be used later 
            _uv = None
            if texVt >= parentUvs[2][1] - zero:
                # use existing vertex
                indexTR = parentIndices[2]
                _uv = parentUvs[2]
                startIndexR = 3
            else:
                aerasbR = False
                indexTR = len(verts)
                verts.append(verts[rs.indexBR] + height*zAxis)
                _uv = (parentItem.uvs[1][0], texVt)

            indicesR = [rs.indexBL, rs.indexBR, indexTR]
            indicesR.extend(reversed(indicesL))
            uvsR = [(parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), _uv]
            # set <_uvIndex> to the last index of <uvsR>
            _uvIndex = 2
            uvsR.extend(reversed(uvsL))
        else:
            indicesR = [rs.indexBL, rs.indexBR, parentIndices[2]]
            uvsR = [(parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), parentUvs[2]]
            startIndexR = 3
            while True:
                if texVt <= parentUvs[startIndexR][1] + zero:
                    if texVt >= parentUvs[startIndexR][1] - zero:
                        # use existing vertex
                        indexTR = parentIndices[startIndexR]
                        uvsR.append(parentUvs[startIndexR])
                        startIndexR += 1
                    else:
                        indexTR = len(verts)
                        # factor
                        k = (parentUvs[startIndexR][1] - texVt) / (texVt - parentUvs[startIndexR-1][1])
                        verts.append(
                            (k*verts[parentIndices[startIndexR-1]] + verts[parentIndices[startIndexR]])/(1.+k)
                        )
                        uvsR.append((
                            (k*parentUvs[startIndexR-1][0] + parentUvs[startIndexR][0])/(1.+k),
                            (k*parentUvs[startIndexR-1][1] + parentUvs[startIndexR][1])/(1.+k)
                        ))
                    indicesR.append(indexTR)
                    break
                indicesR.append(parentIndices[startIndexR])
                uvsR.append(parentUvs[startIndexR])
                startIndexR += 1
            # set <uvsR> to the last index of <uvsR>
            _uvIndex = len(indicesR)-1
            indicesR.extend(reversed(indicesL))
            uvsR.extend(reversed(uvsL))
        
        if berasbL and berasbR:
            # render <levelGroup> in a rectangular geometry
            self.geometryRectangle._renderLevelGroupRectangle(
                parentItem, levelGroup, levelRenderer, rs, indexTL, indexTR, texVt
            )
            if startIndexL != rs.startIndexL:
                rs.startIndexL = startIndexL
            if startIndexR != rs.startIndexR:
                rs.startIndexR = startIndexR
        else:
            item = levelGroup.item
            if item:
                # Set the geometry for the <levelGroup.item>;
                # the lower part of <self> after a devision can be
                # either a rectange (processed in the if-clause) or <self>
                item.geometry = self
            
            if item and item.markup:
                item.indices = indicesR
                item.uvs = uvsR
                levelRenderer.renderDivs(item, levelGroup)
            else:
                levelRenderer.renderLevelGroup(
                    item or parentItem,
                    levelGroup,
                    indicesR,
                    uvsR
                )
            rs.indexBL = indexTL
            rs.indexBR = indexTR
            rs.texVb = texVt
            rs.startIndexL = startIndexL
            rs.startIndexR = startIndexR
        
        # the condition for a triangle as the remaining geometry
        if len(parentIndices)+startIndexL == startIndexR:
            rs.remainingGeometry = self.geometryTriangle
            rs.uvBL, rs.uvBR = uvsR[_uvIndex+1], uvsR[_uvIndex]
        elif aerasbL or aerasbR:
            rs.remainingGeometry = self.geometryPolygon
            rs.uvBL, rs.uvBR = uvsR[_uvIndex+1], uvsR[_uvIndex]
    
    def renderLastLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        if rs.remainingGeometry:
            rs.remainingGeometry.renderLastLevelGroup(parentItem, levelGroup, levelRenderer, rs)
        else:
            parentIndices = parentItem.indices
            parentUvs = parentItem.uvs
            
            item = levelGroup.item
            if levelGroup.item:
                # Set the geometry for the <group.item>;
                # division of a rectangle can only generate rectangles
                levelGroup.item.geometry = self
            
            indices = [rs.indexBL, rs.indexBR]
            indices.extend( parentIndices[i] for i in range(2, len(parentIndices)) )
            uvs = [ (parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb) ]
            uvs.extend( parentUvs[i] for i in range(2, len(parentIndices)) )
                
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
    
    def getClassUvs(self, texUl, texVb, texUr, texVt, uvs):
        numVerts = len(uvs)
        deltaTexU = texUr - texUl
        deltaTexV = texVt - texVb
        deltaV = max( uvs[i][1] for i in range(2, numVerts) ) - uvs[0][1]
        deltaU = uvs[1][0] - uvs[0][0]
        return ( (texUl, texVb), (texUr, texVb) ) + tuple(
                (
                    texUl + deltaTexU * (uvs[i][0] - uvs[0][0]) / deltaU,
                    texVb + deltaTexV * (uvs[i][1] - uvs[0][1]) / deltaV
                ) for i in range(2, numVerts)
            )
    
    def subtract(self, facade, _facade):
        return
    
    def join(self, facade, _facade):
        return
    
    def getMaxV(self, uvs):
        return max(uvs[i][1] for i in range(3, len(uvs)-1))
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        if rs.remainingGeometry:
            rs.remainingGeometry.renderCladdingAtTop(parentItem, parentRenderer)
        else:
            parentIndices = parentItem.indices
            parentUvs = parentItem.uvs
            
            indices = [rs.indexBL, rs.indexBR]
            indices.extend( parentIndices[i] for i in range(rs.startIndexR, len(parentIndices)+rs.startIndexL+1) )
            uvs = [ (parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb) ]
            uvs.extend( parentUvs[i] for i in range(rs.startIndexR, len(parentIndices)+rs.startIndexL+1) )
            self._renderCladding(parentItem, parentRenderer, indices, uvs)
        
    def offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset):
        PolygonHB._offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset, True, True)
    
    def offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset):
        PolygonHB._offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset, True, True)