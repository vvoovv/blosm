from util import zero, zAxis

from . import Geometry
from .rectangle import RectangleFRA


class TrapezoidRV(Geometry):
    """
    A right-angled trapezoid with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """

    def __init__(self):
        self.geometryRectangle = RectangleFRA()
    
    def getFinalUvs(self, item, numLevelsInFace, numTilesU, numTilesV):
        u = len(item.markup)/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            (u, v),
            (0, v)
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

    def renderLevelGroup(self,
            building, levelGroup, parentItem, levelRenderer, height,
            rs
        ):
            verts = building.verts
            parentIndices = parentItem.indices
            
            # <texUl> and <texUr> are the left and right U-coordinates for the rectangular item
            # to be created out of <parentItem>
            texUl = parentItem.uvs[0][0]
            texUr = parentItem.uvs[1][0]
            
            texVt = rs.texVb + height
            
            footprint = parentItem.footprint
            numLevels = footprint.numLevels
            numRoofLevels = footprint.numRoofLevels
            
            # the largest level index (i.e level number) plus one
            upperIndexPlus1 = (levelGroup.index1 if levelGroup.singleLevel else levelGroup.index2) + 1
            
            if rs.tmpTriangle:
                pass
            elif upperIndexPlus1 < numLevels:
                RectangleFRA.renderLevelGroup(self, building, levelGroup, parentItem, levelRenderer, height, rs)
            elif upperIndexPlus1 == numLevels:
                leftVertLower = parentItem.uvs[3][1] < parentItem.uvs[2][1]
                minHeightVertIndex = 3 if leftVertLower else 2
                texVtMin = parentItem.uvs[minHeightVertIndex][1]
                # check if reached one of the upper vertices of the traprezoid
                if texVt == texVtMin:
                    if leftVertLower:
                        # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
                        # an item with rectangular geometry to be created
                        indexTL = parentIndices[3]
                        indexTR = len(building.verts)
                        verts.append(verts[rs.indexBR] + height*zAxis)
                    else:
                        # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
                        # an item with rectangular geometry to be created
                        indexTL = len(building.verts)
                        verts.append(verts[rs.indexBL] + height*zAxis)
                        indexTR = parentIndices[2]
                    if levelGroup:
                        # we have a rectangle here
                        levelGroup.item.geometry = self.geometryRectangle
                    levelRenderer.renderLevelGroup(
                        building, levelGroup, parentItem,
                        (rs.indexBL, rs.indexBR, indexTR, indexTL),
                        ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
                    )
                    rs.tmpTriangle = True
                    
                    rs.indexBL = indexTL
                    rs.indexBR = indexTR
                    rs.texVb = texVt
                elif texVt < texVtMin:
                    RectangleFRA.renderLevelGroup(self, building, levelGroup, parentItem, levelRenderer, height, rs)
                else:
                    rs.tmpTriangle = True
            else:
                return
    
    def renderLastLevelGroup(self, building, levelGroup, parentItem, levelRenderer, rs):
        return


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
            # Additional vertices can be created inside <_item.getItemRenderer(itemRenderer.itemRenderers).render(..)>,
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
                _item.geometry = self
                _item.getItemRenderer(itemRenderer.itemRenderers).render(
                    _item,
                    _indices + tuple( indices[i] for i in range(stopIndexPlus1, startIndex) ) + (indexLT,),
                    _uvs + tuple( uvs[i] for i in range(stopIndexPlus1, startIndex) ) + ((texUl, texVlt),)
                )
            else:
                _item.geometry = self.geometryTrapezoid
                _item.getItemRenderer(itemRenderer.itemRenderers).render(
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
    
    def renderLevelGroup(self,
            building, levelGroup, parentItem, levelRenderer, height,
            rs
        ):
        return
    
    def renderLastLevelGroup(self, building, levelGroup, parentItem, levelRenderer, rs):
        return
    
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