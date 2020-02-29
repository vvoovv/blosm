from util import zero, zAxis

from . import Geometry


class TrapezoidRV(Geometry):
    """
    A right-angled trapezoid with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """
    
    def getFinalUvs(self, item, numLevelsInFace, numTilesU, numTilesV):
        u = len(item.markup)/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            (u, v),
            (0, v)
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
            # Additional vertices can be created inside <itemRenderer.getMarkupItemRenderer(_item).render(..)>,
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
            itemRenderer.getMarkupItemRenderer(_item).render(
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
        itemRenderer.getMarkupItemRenderer(lastItem).render(
            lastItem,
            (rs.indexLB, parentIndices[1], parentIndices[2], rs.indexLT),
            ( (rs.texUl, texVb), (texUr, texVb), (texUr, parentItem.uvs[2][1]), (rs.texUl, rs.texVlt) )
        )

    def renderLevelGroup(self,
            building, levelGroup, parentItem, levelRenderer, height,
            rs
        ):
            verts = building.verts
            # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
            # an item with rectangular geometry to be created
            indexTL = len(building.verts)
            indexTR = indexTL + 1
            # <texUl> and <texUr> are the left and right U-coordinates for the rectangular item
            # to be created out of <parentItem>
            texUl = parentItem.uvs[0][0]
            texUr = parentItem.uvs[1][0]
            verts.append(verts[rs.indexBL] + height*zAxis)
            verts.append(verts[rs.indexBR] + height*zAxis)
            texVt = rs.texVb + height
            if levelGroup:
                # Set the geometry for the <levelGroup.item>; division of a rectangle can only generate rectangles
                levelGroup.item.geometry = self
            levelRenderer.renderLevelGroup(
                building, levelGroup, parentItem,
                (rs.indexBL, rs.indexBR, indexTR, indexTL),
                ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
            )
            rs.indexBL = indexTL
            rs.indexBR = indexTR
            rs.texVb = texVt


class TrapezoidChainedRV(Geometry):
    """
    A sequence of adjoining right-angled trapezoids with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """
    
    def __init__(self):
        self.geometryTrapezoid = TrapezoidRV()
    
    def initRenderStateForDivs(self, rs, item):
        super().initRenderStateForDivs(rs, item)
        rs.startIndex = len(item.indices) - 1
    
    def renderDivs(self,
            itemRenderer, building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            rs
        ):
        # <startIndex> is used for optimization
        verts = building.verts
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
            # Additional vertices can be created inside <itemRenderer.getMarkupItemRenderer(_item).render(..)>,
            # that's why we use <len(building.verts)>
            indexRB = len(building.verts)
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
                item.geometry = self
                itemRenderer.getMarkupItemRenderer(_item).render(
                    _item,
                    _indices + tuple( indices[i] for i in range(stopIndexPlus1, startIndex) ) + (indexLT,),
                    _uvs + tuple( uvs[i] for i in range(stopIndexPlus1, startIndex) ) + ((texUl, texVlt),)
                )
            else:
                _item.geometry = self.geometryTrapezoid
                itemRenderer.getMarkupItemRenderer(_item).render(
                    _item,
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
        # Set the geometry for the <lastItem>; division of a trapezoid can only generate trapezoids
        lastItem.geometry = self
        chainedTrapezoid = rs.startIndex > 3
        itemRenderer.getMarkupItemRenderer(lastItem).render(
            lastItem,
            # indices
            (rs.indexLB, parentIndices[1], parentIndices[2]) +\
                tuple( parentIndices[i] for i in range(3, rs.startIndex) ) +\
                (rs.indexLT,)\
            if chainedTrapezoid else\
            (rs.indexLB, parentIndices[1], parentIndices[2], rs.indexLT),
            # UV-coordinates
            ( (rs.texUl, texVb), (texUr, texVb), (texUr, parenItem.uvs[2][1]) ) +\
                tuple( uvs[i] for i in range(3, rs.startIndex) ) +\
                ((rs.texUl, rs.texVlt),)
            if chainedTrapezoid else\
            ( (rs.texUl, texVb), (texUr, texVb), (texUr, parenItem.uvs[2][1]), (rs.texUl, rs.texVlt) )
        )