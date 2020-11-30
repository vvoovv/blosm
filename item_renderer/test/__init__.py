

class GeometryRenderer:

    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
    
    def render(self, roofItem):
        self.r.app.log.write("Not hipped roof:%s\n" % roofItem.building.outline.tags["id"])
        

class GeometryRendererRoofFlat(GeometryRenderer):
    
    def render(self, roofItem):
        self.r.app.log.write("Flat roof:%s\n" % roofItem.building.outline.tags["id"])


class GeometryRendererRoofWithSides(GeometryRenderer):
    
    def render(self, roofItem):
        if roofItem.exception:
            self.r.app.log.write("%s:%s\n" % (roofItem.exception, roofItem.building.outline.tags["id"]))
            return
        # check for faces consisting of less than 3 vertices
        for roofSide in roofItem.roofSides:
            if len(roofSide.indices) < 3:
                self.r.app.log.write("less than 3 verts:%s\n" % roofItem.building.outline.tags["id"])
                break
        for roofSide in roofItem.roofSides:
            if len(roofSide.indices) != len(set(roofSide.indices)):
                self.r.app.log.write("duplicated indices:%s\n" % roofItem.building.outline.tags["id"])
                break
    
    def _render(self, roofItem):        
        for roofSide in roofItem.roofSides:
            self.r.createFace(
                roofItem.building,
                roofSide.indices
            )


class GeometryRendererFacade(GeometryRenderer):
        
    def render(self, footprint, data):
        for facade in footprint.facades:
            self.r.createFace(footprint.building, facade.indices)