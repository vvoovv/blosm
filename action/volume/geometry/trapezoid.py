from util import zero


class TrapezoidRV:
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
    
    def generateDivs(self,
            itemRenderer, building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            indexLB, indexLT, texUl, texVlt,
            startIndex
        ):
        # <startIndex> is not used by the <TrapezoidRV> geometry
        verts = building.verts
        # <texVb> is the V-coordinate for the bottom vertices of the trapezoid items
        # to be created out of <item>
        texVb = item.uvs[0][1]
        v1 = verts[indexLB]
        v2 = verts[indexLT]
        # <factor> is used in the calculations below, it is actually a tangens of the trapezoid angle
        factor = (item.uvs[2][1] - item.uvs[3][1])/item.width
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            # <indexRB> and <indexRT> are indices of the bottom and top vertices
            # on the right side of an item with rectangular geometry to be created
            # Additional vertices can be created inside <itemRenderer.getItemRenderer(_item).render(..)>,
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
            itemRenderer.getItemRenderer(_item).render(
                _item,
                (indexLB, indexRB, indexRT, indexLT),
                ( (texUl, texVb), (texUr, texVb), (texUr, texVrt), (texUl, texVlt) )
            )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
            texVlt = texVrt
        return indexLB, indexLT, texUl, texVlt, startIndex

    def generateLastDiv(self, itemRenderer, item, lastItem, indexLB, indexLT, texUl, texVlt, startIndex):
        # <startIndex> is not used by the <TrapezoidRV> geometry
        parentIndices = item.indices
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <item>
        texUr = item.uvs[1][0]
        # <texVb> is the V-coordinate for bottom vertices of the trapezoid item
        # to be created out of <item>
        texVb = item.uvs[0][1]
        # Set the geometry for the <lastItem>; division of a rectangle can only generate rectangles
        lastItem.geometry = self
        itemRenderer.getItemRenderer(lastItem).render(
            lastItem,
            (indexLB, parentIndices[1], parentIndices[2], indexLT),
            ( (texUl, texVb), (texUr, texVb), (texUr, item.uvs[2][1]), (texUl, texVlt) )
        )


class TrapezoidChainedRV:
    """
    A sequence of adjoining right-angled trapezoids with the right angles at the bottom side and
    parallel sides along the vertical (z) axis
    """
    def generateDivs(self,
            itemRenderer, building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            indexLB, indexLT, texUl, texVlt,
            startIndex
        ):
        # <startIndex> is used for optimization
        verts = building.verts
        indices = item.indices
        uvs = item.uvs
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
            # Additional vertices can be created inside <itemRenderer.getItemRenderer(_item).render(..)>,
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
            itemRenderer.getItemRenderer(_item).render(
                _item,
                _indices + tuple( indices[i] for i in range(stopIndexPlus1, startIndex) ) + (indexLT,)\
                    if chainedTrapezoid else\
                    (indexLB, indexRB, indexRT, indexLT),
                _uvs + tuple( uvs[i] for i in range(stopIndexPlus1, startIndex) ) + ((texUl, texVlt),)
                    if chainedTrapezoid else\
                    ( (texUl, texVb), (texUr, texVb), (texUr, texVrt), (texUl,texVlt) )
            )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
            texVlt = texVrt
            startIndex = stopIndexPlus1
        return indexLB, indexLT, texUl, texVlt, startIndex
    
    def generateLastDiv(self, itemRenderer, item, lastItem, indexLB, indexLT, texUl, texVlt, startIndex):
        parentIndices = item.indices
        uvs = item.uvs
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <item>
        texUr = item.uvs[1][0]
        # <texVb> is the V-coordinate for bottom vertices of the trapezoid item
        # to be created out of <item>
        texVb = item.uvs[0][1]
        # Set the geometry for the <lastItem>; division of a rectangle can only generate rectangles
        lastItem.geometry = self
        chainedTrapezoid = startIndex > 3
        itemRenderer.getItemRenderer(lastItem).render(
            lastItem,
            # indices
            (indexLB, parentIndices[1], parentIndices[2]) +\
                tuple( parentIndices[i] for i in range(3, startIndex) ) +\
                (indexLT,)\
            if chainedTrapezoid else\
            (indexLB, parentIndices[1], parentIndices[2], indexLT),
            # UV-coordinates
            ( (texUl, texVb), (texUr, texVb), (texUr, item.uvs[2][1]) ) +\
                tuple( uvs[i] for i in range(3, startIndex) ) +\
                ((texUl, texVlt),)
            if chainedTrapezoid else\
            ( (texUl, texVb), (texUr, texVb), (texUr, item.uvs[2][1]), (texUl, texVlt) )
        )