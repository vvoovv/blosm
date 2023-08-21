from . import Geometry


class Triangle(Geometry):
    
    def __init__(self, geometryTrapezoid):
        # a classical trapezoid with the sides parallel to the horizontal axis
        self.geometryTrapezoid = geometryTrapezoid
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            ( (itemUvs[2][0] - itemUvs[0][0]) / (itemUvs[1][0] - itemUvs[0][0]) * u, v)
        )
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        # <rs.indices> and <rs.uvs> are used if <self.renderLastLevelGroup(..)> is called
        # from another geometry where <self> is the remaining geometry
        parentIndices = rs.indices or parentItem.indices
        parentUvs = rs.uvs or parentItem.uvs
        
        verts = parentItem.building.renderInfo.verts
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        texVt = rs.texVb + height
        # <indexTL> and <indexTR> are indices of the left and right vertices on the top side of
        # an item with the trapezoid geometry to be created
        indexTL = len(verts)
        kL = (parentUvs[2][1] - texVt) / (texVt - parentUvs[0][1])
        verts.append( ( kL*verts[parentIndices[0]] + verts[parentIndices[2]])/(1.+kL) )
        
        indexTR = indexTL + 1
        kR = (parentUvs[2][1] - texVt) / (texVt - parentUvs[1][1])
        verts.append( ( kR*verts[parentIndices[1]] + verts[parentIndices[2]])/(1.+kR) )
        
        indices = (rs.indexBL, rs.indexBR, indexTR, indexTL)
        uvs = (
            parentUvs[0],
            parentUvs[1],
            ( (kR*parentUvs[1][0]+parentUvs[2][0])/(1.+kR), texVt ),
            ( (kL*parentUvs[0][0]+parentUvs[2][0])/(1.+kL), texVt )
        )
        
        item = levelGroup.item
        if item:
            # Set the geometry for the <levelGroup.item>;
            # division of a triangle can only generate a classical trapezoid at the bottom and
            # a triangle at the top
            item.geometry = self.geometryTrapezoid
        
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
        if rs.remainingGeometry:
            # update <rs.indices> and <rs.uvs>
            rs.indices = (indexTL, indexTR, parentIndices[2])
            rs.uvs = (uvs[3], uvs[2], parentUvs[2])
    
    def renderLastLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        # <rs.indices> and <rs.uvs> are used if <self.renderLastLevelGroup(..)> is called
        # from another geometry where <self> is the remaining geometry
        parentIndices = rs.indices or parentItem.indices
        parentUvs = rs.uvs or parentItem.uvs
        
        item = levelGroup.item
        if levelGroup.item:
            levelGroup.item.geometry = self
            
        if item and item.markup:
            item.indices = (rs.indexBL, rs.indexBR, parentIndices[2])
            item.uvs = ( (parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), parentUvs[2] )
            levelRenderer.renderDivs(item, levelGroup)
        else:
            levelRenderer.renderLevelGroup(
                item or parentItem,
                levelGroup,
                (rs.indexBL, rs.indexBR, parentIndices[2]),
                ( (parentUvs[0][0], rs.texVb), (parentUvs[1][0], rs.texVb), parentUvs[2] )
            )
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        parentRenderer.renderCladding(
            parentItem,
            parentRenderer.r.createFace(
                parentItem.footprint,
                rs.indices or (rs.indexBL, rs.indexBR, parentItem.indices[2])
            ),
            rs.uvs or ( (parentItem.uvs[0][0], rs.texVb), (parentItem.uvs[1][0], rs.texVb), parentItem.uvs[2] )
        )