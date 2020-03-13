

class Geometry:

    def initRenderStateForDivs(self, rs, item):
        rs.indexLB = item.indices[0]
        rs.indexLT = item.indices[-1]
        rs.texUl = item.uvs[0][0]
        rs.texVlt = item.uvs[-1][1]

    def initRenderStateForLevels(self, rs, parentItem):
        parentIndices = parentItem.indices
        # <indexBL> and <indexBR> are indices of the bottom vertices of an level item to be created
        # The prefix <BL> means "bottom left"
        rs.indexBL = parentIndices[0]
        # The prefix <BR> means "bottom rights"
        rs.indexBR = parentIndices[1]
        # <texVb> is the current V-coordinate for texturing the bottom vertices of
        # level items to be created out of <parentItem>
        rs.texVb = parentItem.uvs[0][1]
    
    def renderBottom(self, parentItem, parentRenderer, rs):
        bottom = parentItem.levelGroups.bottom
        if bottom:
            self.renderLevelGroup(
                parentItem, bottom, parentRenderer.bottomRenderer, rs
            )