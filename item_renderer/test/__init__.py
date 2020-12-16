from parse.osm.way import Way


class GeometryRenderer:

    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
    
    def writeLog(self, roofItem, message):
        self.r.app.log.write("%s|%s|%s\n" %
            (
                roofItem.building.outline.tags["id"],
                "way" if isinstance(roofItem.building.outline, Way) else "relation",
                message
            )
        )
    
    def render(self, roofItem):
        self.writeLog(roofItem, "Not hipped roof")


class GeometryRendererRoofFlat(GeometryRenderer):
    
    def render(self, roofItem):
        self.writeLog(roofItem, "Flat roof")


class GeometryRendererRoofWithSides(GeometryRenderer):
    
    def render(self, roofItem):
        if roofItem.exception:
            self.writeLog(roofItem, roofItem.exception)
            return
        # check for faces consisting of less than 3 vertices
        for roofSide in roofItem.roofSides:
            if len(roofSide.indices) < 3:
                self.writeLog(roofItem, "Less than 3 verts")
                break
        for roofSide in roofItem.roofSides:
            if len(roofSide.indices) != len(set(roofSide.indices)):
                self.writeLog(roofItem, "Duplicated indices")
                break
        #self.writeLog(roofItem, "Ok")


class GeometryRendererFacade(GeometryRenderer):
        
    def render(self, footprint, data):
        return