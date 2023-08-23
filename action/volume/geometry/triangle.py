from . import Geometry


class Triangle(Geometry):
    
    def __init__(self, geometryTrapezoid):
        # a classical trapezoid with the sides parallel to the horizontal axis
        self.geometryTrapezoid = geometryTrapezoid
    
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