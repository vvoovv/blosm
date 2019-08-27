from mathutils import Vector
from grammar.arrangement import Horizontal, Vertical


class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            self.renderLevels(item)
        else:
            self.renderDivs(item)
    
    def renderLevels(self, item):
        pass
    
    def renderDivs(self, item):
        # <r> is the global building renderer
        r = self.r
        building = item.building
        verts = building.verts
        indexOffset = len(verts)
        parentIndices = item.indices
        
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            item.calculateMarkupDivision()
            # create vertices for the markup items
            numItems = len(item.markup)
            if numItems == 1:
                # the special case
                _item = item.markup[0]
                _item.indices = parentIndices
                _item.uvs = item.uvs
            else:
                building.appendBmVerts(2*(numItems-1))
                _item = item.markup[0]
                prevIndex1 = parentIndices[0]
                prevIndex2 = parentIndices[3]
                v1 = verts[prevIndex1]
                v2 = verts[prevIndex2]
                unitVector = (verts[parentIndices[1]] - v1) / item.width
                z1 = v1[2]
                z2 = v2[2]
                _item.indices = (prevIndex1, indexOffset, indexOffset+1, prevIndex2)
                index1 = indexOffset
                index2 = indexOffset + 1
                # process all items but the last one
                for _i in range(len(item.markup)-1):
                    _item = item.markup[_i]
                    incrementVector = _item.width * unitVector
                    v1 += incrementVector
                    verts.append(v1)
                    v2 += incrementVector
                    verts.append(v2)
                    _item.indices = (prevIndex1, index1, index2, prevIndex2)
                    prevIndex1 = index1
                    prevIndex2 = index2
                    index1 += 2
                    index2 += 2
                    r.createFace(building, _item.indices, None)
                # process the last item
                _item = item.markup[-1]
                _item.indices = (prevIndex1, parentIndices[1], parentIndices[2], prevIndex2)
                r.createFace(building, _item.indices, None)
        else:
            pass
        