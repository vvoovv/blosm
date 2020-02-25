from util import zAxis


class Rectangle:
        
    def getUvs(self, width, height):
        """
        Get flat vertices coordinates on the facade surface (i.e. on the rectangle)
        """
        return ( (0., 0.), (width, 0.), (width, height), (0., height) )
    
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
            indexLB, indexLT, texUl, texVt,
            startIndex
        ):
        # <startIndex> is not used by the <Rectangle> geometry
        verts = building.verts
        # <texVb> is the V-coordinate for the bottom vertices of the rectangular items
        # to be created out of <item>
        texVb = item.uvs[0][1]
        v1 = verts[indexLB]
        v2 = verts[indexLT]
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            # Set the geometry for the <_item>; division of a rectangle can only generate rectangles
            _item.geometry = self
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
            verts.append(v2)
            texUr = texUl + _item.width
            itemRenderer.getItemRenderer(_item).render(
                _item,
                (indexLB, indexRB, indexRT, indexLT),
                ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
            )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
        return indexLB, indexLT, texUl, texVt, startIndex
    
    def generateLastDiv(self, itemRenderer, item, lastItem, indexLB, indexLT, texUl, texVt, startIndex):
        # <texVt> and <startIndex> are not used by the <Rectangle> geometry
        parentIndices = item.indices
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <item>
        texUr = item.uvs[1][0]
        # <texVb> and <texVt> are the bottom (b) and top (t) V-coordinates for the rectangular item
        # to be created out of <item>
        texVb = item.uvs[0][1]
        texVt = item.uvs[3][1]
        # Set the geometry for the <lastItem>; division of a rectangle can only generate rectangles
        lastItem.geometry = self
        itemRenderer.getItemRenderer(lastItem).render(
            lastItem,
            (indexLB, parentIndices[1], parentIndices[2], indexLT),
            ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
        )
    
    def generateLevelDiv(self,
            building, levelGroup, parentItem, renderer, height,
            indexBL, indexBR, texVb
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
            verts.append(verts[indexBL] + height*zAxis)
            verts.append(verts[indexBR] + height*zAxis)
            texVt = texVb + height
            if levelGroup:
                # Set the geometry for the <levelGroup.item>; division of a rectangle can only generate rectangles
                levelGroup.item.geometry = self
            renderer.renderLevelGroup(
                building, levelGroup, parentItem,
                (indexBL, indexBR, indexTR, indexTL),
                ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
            )
            return indexTL, indexTR, texVt
    
    def generateLastLevelDiv(self, itemRenderer, building, group, parentItem, indexBL, indexBR, texVb):
        parentIndices = parentItem.indices
        texVt = parentItem.uvs[2][1]
        # <texUl> and <texUr> are the left and right U-coordinates for the rectangular items
        # to be created out of <parentItem>
        texUl = parentItem.uvs[0][0]
        texUr = parentItem.uvs[1][0]
        # Set the geometry for the <group.item>; division of a rectangle can only generate rectangles
        group.item.geometry = self
        itemRenderer.levelRenderer.getRenderer(group).renderLevelGroup(
            building, group, parentItem,
            (indexBL, indexBR, parentIndices[2], parentIndices[3]),
            ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
        )
        