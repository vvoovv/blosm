

class Geometry:

    def initRenderStateForDivs(self, rs, item):
        rs.indexLB = item.indices[0]
        rs.indexLT = item.indices[-1]
        rs.texUl = item.uvs[0][0]
        rs.texVlt = item.uvs[-1][1]