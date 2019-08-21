from .container import Container


class Facade(Container):
    
    def __init__(self):
        super().__init__()
    
    def render(self, footprint):
        # <r> is the global building renderer
        r = self.r
        building = footprint.building
        
        facadeStyle = footprint.facadeStyle
        if footprint.facadeStyle:
            for facade in footprint.facades:
                for styleBlock in facadeStyle:
                    if facade.evaluateCondition(styleBlock):
                        facade.styleBlock = styleBlock
                        # check if <minWidth> attribute is given in the styleBlock
                        minWidth = facade.getStyleBlockAttr("minWidth")
                        if (minWidth and facade.width > minWidth) or not minWidth:
                            if styleBlock.markup:
                                self.renderMarkup(facade)
                                if facade.valid:
                                    break
                                else:
                                    # <styleBlock> does not suit for <facade>
                                    # Make <facade> valid again to try it
                                    # with the next <styleBlock> from <facadeStyle>
                                    facade.valid = True
                            else:
                                pass
                        # Clean up the styleBlock for the next attempt with
                        # the next style block from <facadeStyle>
                        facade.styleBlock = None
        else:
            # simply create BMFaces here
            for facade in footprint.facades:
                r.createFace(building, facade.indices, facade.uvs)