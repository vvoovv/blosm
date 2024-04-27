from .section import Section


class SideLane(Section):
    
    def renderItem(self, section):
        createPolylineMesh(None, section.street.bm, section.centerline)
    
    def finalizeItem(self, section, itemIndex):
        self.setModifierSection(section, itemIndex, 0., 0.)
        
        #
        # set the index of the street section
        #
        obj = section.street.obj
        for pointIndex in range(self.pointIndexOffset, self.pointIndexOffset + len(section.centerline)):
            obj.data.attributes['section_index'].data[pointIndex].value = itemIndex
        
        self.pointIndexOffset += len(section.centerline)