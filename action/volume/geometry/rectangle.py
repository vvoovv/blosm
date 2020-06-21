from util import zAxis

from . import Geometry


class RectangleFRA(Geometry):
    """
    Rectangle for Flat Roofs and Alike (Roofs based on a generatrix)
    """
        
    def getUvs(self, width, height):
        """
        Get flat vertices coordinates on the facade surface (i.e. on the rectangle)
        """
        return ( (0., 0.), (width, 0.), (width, height), (0., height) )
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV):
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            (u, v),
            (0, v)
        )
    
    def getClassUvs(self, texUl, texVb, texUr, texVt, uvs):
        return (
            (texUl, texVb),
            (texUr, texVb),
            (texUr, texVt),
            (texUl, texVt)
        )
    
    def renderDivs(self,
            itemRenderer, building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            rs
        ):
        # <startIndex> is not used by the <Rectangle> geometry
        verts = building.verts
        # <texVb> is the V-coordinate for the bottom vertices of the rectangular items
        # to be created out of <item>
        texVb = item.uvs[0][1]
        indexLB = rs.indexLB
        v1 = verts[indexLB]
        indexLT = rs.indexLT
        v2 = verts[indexLT]
        
        texUl = rs.texUl
        texVt = rs.texVlt
        
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            # Set the geometry for the <_item>; division of a rectangle can only generate rectangles
            _item.geometry = self
            # <indexRB> and <indexRT> are indices of the bottom and top vertices
            # on the right side of an item with rectangular geometry to be created
            # Additional vertices can be created inside <_item.getItemRenderer(item.itemRenderers).render(..)>,
            # that's why we use <len(building.verts)>
            indexRB = len(building.verts)
            indexRT = indexRB + 1
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            v2 = v2 + incrementVector
            verts.append(v2)
            texUr = texUl + _item.width
            _item.getItemRenderer(itemRenderer.itemRenderers).render(
                _item,
                (indexLB, indexRB, indexRT, indexLT),
                ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
            )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
        
        rs.indexLB = indexLB
        rs.indexLT = indexLT
        rs.texUl = texUl
    
    def renderLastDiv(self, itemRenderer, parentItem, lastItem, rs):
        # <texVt> and <startIndex> are not used by the <Rectangle> geometry
        parentIndices = parentItem.indices
        indexLB = rs.indexLB
        indexLT = rs.indexLT
        texUl = rs.texUl
        # <texUr> is the right U-coordinate for the rectangular item to be created out of <parentItem>
        texUr = parentItem.uvs[1][0]
        # <texVb> and <texVt> are the bottom (b) and top (t) V-coordinates for the rectangular item
        # to be created out of <parentItem>
        texVb = parentItem.uvs[0][1]
        texVt = parentItem.uvs[3][1]
        # Set the geometry for the <lastItem>; division of a rectangle can only generate rectangles
        lastItem.geometry = self
        lastItem.getItemRenderer(itemRenderer.itemRenderers).render(
            lastItem,
            (indexLB, parentIndices[1], parentIndices[2], indexLT),
            ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
        )
    
    def renderLevelGroups(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        self.initRenderStateForLevels(rs, parentItem)
        self.renderBottom(parentItem, parentRenderer, rs)
        
        levelGroups = parentItem.levelGroups
        
        if levelGroups.begin.next:
            # There are at least 2 level groups or one level group and the top
            # Detach the end level group. It can be either the top or the last level group
            lastLevelGroup = levelGroups.end
            lastLevelGroup.prev.next = None
            
            levelGroup = levelGroups.begin
            while levelGroup:
                self.renderLevelGroup(
                    parentItem, 
                    levelGroup,
                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers),
                    rs
                )
                levelGroup = levelGroup.next
        else:
            lastLevelGroup = levelGroups.begin
        
        self.renderLastLevelGroup(
            parentItem, 
            lastLevelGroup,
            lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers)\
                if lastLevelGroup.item else parentRenderer,
            rs
        )
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
            building = parentItem.building
            verts = building.verts
            height = levelGroup.levelHeight\
                if levelGroup.singleLevel else\
                levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
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
            if levelGroup.item:
                # Set the geometry for the <levelGroup.item>;
                # division of a rectangle can only generate rectangles
                levelGroup.item.geometry = self
            levelRenderer.renderLevelGroup(
                parentItem,
                levelGroup,
                (rs.indexBL, rs.indexBR, indexTR, indexTL),
                ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
            )
            rs.indexBL = indexTL
            rs.indexBR = indexTR
            rs.texVb = texVt
    
    def renderLastLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        parentIndices = parentItem.indices
        texVt = parentItem.uvs[2][1]
        # <texUl> and <texUr> are the left and right U-coordinates for the rectangular items
        # to be created out of <parentItem>
        texUl = parentItem.uvs[0][0]
        texUr = parentItem.uvs[1][0]
        if levelGroup.item:
            # Set the geometry for the <group.item>;
            # division of a rectangle can only generate rectangles
            levelGroup.item.geometry = self
        levelRenderer.renderLevelGroup(
            parentItem,
            levelGroup,
            (rs.indexBL, rs.indexBR, parentIndices[2], parentIndices[3]),
            ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
        )
        