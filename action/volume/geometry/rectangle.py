from math import floor
from mathutils import Vector
from util import zAxis

from . import Geometry


def _updateFacadeBottomVerts(facade, z, verts):
    numVerts = len(verts)
    indices = facade.indices
    # add new vertice for the bottom part of the facade
    verts.append( Vector( (verts[indices[0]][0], verts[indices[0]][1], z)) )
    verts.append( Vector((verts[indices[1]][0], verts[indices[1]][1], z)) )
    
    facade.indices = (numVerts, numVerts+1, indices[2], indices[3])
    
    facade.minHeight = z
    
    minLevel = facade.footprint.levelHeights.calculateMinLevelFromMinHeight(z)
    if minLevel >= facade.footprint.numLevels:
        facade.highEnoughForLevel = False
    else:
        facade.minLevel = minLevel


def _updateFacadeTopVerts(facade, z, verts):
    numVerts = len(verts)
    indices = facade.indices
    # add new vertice for the bottom part of the facade
    verts.append( Vector( (verts[indices[2]][0], verts[indices[2]][1], z)) )
    verts.append( Vector((verts[indices[3]][0], verts[indices[3]][1], z)) )
    
    facade.indices = (indices[0], indices[1], numVerts, numVerts+1)


class RectangleFRA(Geometry):
    """
    Rectangle for Flat Roofs and Alike (Roofs based on a generatrix)
    """
        
    def getUvs(self, width, height):
        """
        Get flat vertices coordinates on the facade surface (i.e. on the rectangle)
        """
        return ( (0., 0.), (width, 0.), (width, height), (0., height) )
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, _):
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
            itemRenderer, item, levelGroup, markupItemIndex1, markupItemIndex2, step,
            rs
        ):
        # If <levelGroup> is given, that actually means that <item> is a level or contained
        # inside another level item. In this case the call to <self.renderLevelGroup(..)>
        # will be made later in the code
        
        unitVector = item.facade.vector.unitVector3d
        # <startIndex> is not used by the <Rectangle> geometry
        verts = item.building.renderInfo.verts
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
            if not _item.width:
                continue
            # Set the geometry for the <_item>; division of a rectangle can only generate rectangles
            _item.geometry = self
            # <indexRB> and <indexRT> are indices of the bottom and top vertices
            # on the right side of an item with rectangular geometry to be created
            # Additional vertices can be created inside <_item.getItemRenderer(item.itemRenderers).render(..)>,
            # that's why we use <len(verts)>
            indexRB = len(verts)
            indexRT = indexRB + 1
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            v2 = v2 + incrementVector
            verts.append(v2)
            texUr = texUl + _item.width
            if levelGroup:
                _item.getItemRenderer(itemRenderer.itemRenderers).renderLevelGroup(
                    _item,
                    levelGroup,
                    (indexLB, indexRB, indexRT, indexLT),
                    ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
                )
            indexLB = indexRB
            indexLT = indexRT
            texUl = texUr
        
        rs.indexLB = indexLB
        rs.indexLT = indexLT
        rs.texUl = texUl
    
    def renderLastDiv(self, itemRenderer, parentItem, levelGroup, lastItem, rs):
        # If <levelGroup> is given, that actually means that <item> is a level or contained
        # inside another level item. In this case the call to <self.renderLevelGroup(..)>
        # will be made later in the code
        
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
        lastItem.getItemRenderer(itemRenderer.itemRenderers).renderLevelGroup(
            lastItem,
            levelGroup,
            (indexLB, parentIndices[1], parentIndices[2], indexLT),
            ( (texUl, texVb), (texUr, texVb), (texUr, texVt), (texUl, texVt) )
        )
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        verts = parentItem.building.renderInfo.verts
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
        # an item with rectangular geometry to be created
        indexTL = len(verts)
        indexTR = indexTL + 1
        verts.append(verts[rs.indexBL] + height*zAxis)
        verts.append(verts[rs.indexBR] + height*zAxis)
        texVt = rs.texVb + height
        
        self._renderLevelGroupRectangle(parentItem, levelGroup, levelRenderer, rs, indexTL, indexTR, texVt)
    
    def _renderLevelGroupRectangle(self, parentItem, levelGroup, levelRenderer, rs, indexTL, indexTR, texVt):
        # <texUl> and <texUr> are the left and right U-coordinates for the rectangular item
        # to be created out of <parentItem>
        texUl = parentItem.uvs[0][0]
        texUr = parentItem.uvs[1][0]
               
        item = levelGroup.item
        if item:
            # Set the geometry for the <levelGroup.item>;
            # division of a rectangle can only generate rectangles
            item.geometry = self
        
        if item and item.markup:
            item.indices = (rs.indexBL, rs.indexBR, indexTR, indexTL)
            item.uvs = ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
            levelRenderer.renderDivs(item, levelGroup)
        else:
            levelRenderer.renderLevelGroup(
                item or parentItem,
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
    
    def join(self, facade, _facade):
        # <self> is the geometry for <facade>
        
        # <facade> and <_facade> belongs to the same building
        verts = facade.footprint.building.renderInfo.verts
        
        _geometry = _facade.geometry
        indices, _indices = facade.indices, _facade.indices
        
        if isinstance(_geometry, RectangleFRA):
            z1, z2 = verts[indices[0]][2], verts[indices[-1]][2]
            _z1, _z2 = verts[_indices[0]][2], verts[_indices[-1]][2]
            
            if z1 <= _z1 < _z2 <= z2:
                _facade.visible = False
            elif _z1 <= z1 < z2 <= _z2:
                facade.visible = False
            elif z1 < _z1 < z2:
                _updateFacadeBottomVerts(_facade, z2, verts)
            elif _z1 < z1 < _z2:
                _updateFacadeBottomVerts(facade, _z2, verts)
    
    def subtract(self, facade, _facade):
        # <self> is the geometry for <facade>
        
        verts, _verts = facade.footprint.building.renderInfo.verts, _facade.footprint.building.renderInfo.verts
        _geometry = _facade.geometry
        indices, _indices = facade.indices, _facade.indices
        
        if isinstance(_geometry, RectangleFRA):
            z1, z2 = verts[indices[0]][2], verts[indices[-1]][2]
            _z1, _z2 = _verts[_indices[0]][2], _verts[_indices[-1]][2]
            
            if z1 == _z1:
                if _z2 < z2:
                    _facade.visible = False
                    _updateFacadeBottomVerts(facade, _z2, verts)
                elif z2 < _z2:
                    facade.visible = False
                    _updateFacadeBottomVerts(_facade, z2, _verts)
                else:
                    # z2 == _z2
                    facade.visible = _facade.visible = False
            elif z2 == _z2:
                if z1 < _z1:
                    _facade.visible = False
                    _updateFacadeTopVerts(facade, _z1, verts)
                elif _z1 < z1:
                    facade.visible = False
                    _updateFacadeTopVerts(_facade, z1, _verts)
            elif z1 < _z1 < z2 < _z2:
                _updateFacadeBottomVerts(_facade, z2, _verts)
                _updateFacadeTopVerts(facade, _z1, verts)
            elif _z1 < z1 < _z2 < z2:
                _updateFacadeBottomVerts(facade, _z2, verts)
                _updateFacadeTopVerts(_facade, z1, _verts)
            elif z1 < _z1 < _z2 < z2:
                _facade.visible = False
            elif _z1 < z1 < z2 < _z2:
                facade.visible = False
    
    def getMaxV(self, uvs):
        return uvs[2][1]
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        texVt = parentItem.uvs[2][1]
        # <texUl> and <texUr> are the left and right U-coordinates for the rectangular items
        # to be created out of <parentItem>
        texUl = parentItem.uvs[0][0]
        texUr = parentItem.uvs[1][0]
        self._renderCladding(
            parentItem,
            parentRenderer,
            (rs.indexBL, rs.indexBR, parentItem.indices[2], parentItem.indices[3]),
            ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
        )
    
    def renderCladdingStripAtMiddle(self, height, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        verts = parentItem.building.renderInfo.verts
        
        # <indexTL> and <indexTR> are the indices of the left and right vertices on the top side of
        # the rectangular cladding strip
        indexTL = len(verts)
        indexTR = indexTL + 1
        # <texUl> and <texUr> are the left and right U-coordinates of the the rectangular cladding strip
        texUl = parentItem.uvs[0][0]
        texUr = parentItem.uvs[1][0]
        verts.append(verts[rs.indexBL] + height*zAxis)
        verts.append(verts[rs.indexBR] + height*zAxis)
        texVt = rs.texVb + height
        
        self._renderCladding(
            parentItem,
            parentRenderer,
            (rs.indexBL, rs.indexBR, indexTR, indexTL),
            ( (texUl, rs.texVb), (texUr, rs.texVb), (texUr, texVt), (texUl, texVt) )
        )
        
        rs.indexBL = indexTL
        rs.indexBR = indexTR
        rs.texVb = texVt
    
    def offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset):
        verts = item.building.renderInfo.verts
        offsetVec = offset/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        
        # the new vertex at the bottom
        indexB = len(verts)
        verts.append( verts[parentIndices[0]] + offsetVec )
        uvB = (parentUvs[0][0] + offset, parentUvs[0][1])
        
        # the new vertex at the top
        indexT = indexB + 1
        verts.append( verts[parentIndices[3]] + offsetVec )
        uvT = (parentUvs[3][0] + offset, parentUvs[3][1])
        
        item.indices = (indexB, parentIndices[1], parentIndices[2], indexT)
        item.uvs = (uvB, parentUvs[1], parentUvs[2], uvT)
        
        # render offset area
        self._renderCladding(
            item,
            renderer,
            (parentIndices[0], indexB, indexT, parentIndices[3]),
            (parentUvs[0], uvB, uvT, parentUvs[3])
        )
    
    def offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset):
        verts = item.building.renderInfo.verts
        offsetVec = (parentUvs[1][0] - parentUvs[0][0] - offset) / (parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        
        # the new vertex at the bottom
        indexB = len(verts)
        verts.append( verts[parentIndices[0]] + offsetVec )
        uvB = (parentUvs[0][0] + offset, parentUvs[0][1])
        
        # the new vertex at the top
        indexT = indexB + 1
        verts.append( verts[parentIndices[3]] + offsetVec )
        uvT = (parentUvs[3][0] + offset, parentUvs[3][1])
        
        item.indices = (parentIndices[0], indexB, indexT, parentIndices[3])
        item.uvs = (parentUvs[0], uvB, uvT, parentUvs[3])
        
        # render offset area
        self._renderCladding(
            item,
            renderer,
            (indexB, parentIndices[1], parentIndices[2], indexT),
            (uvB, parentUvs[1], parentUvs[2], uvT)
        )