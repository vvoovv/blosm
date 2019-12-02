from .container import Container


class Facade(Container):
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
        self.levelRenderer = itemRenderers["Level"]
        self.basementRenderer = itemRenderers["Basement"]
    
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
                                    facade.markup.clear()
                            else:
                                pass
                        # Clean up the styleBlock for the next attempt with
                        # the next style block from <facadeStyle>
                        facade.styleBlock = None
                else:
                    # no style block suits the <facade>
                    r.createFace(building, facade.indices)
        else:
            # simply create BMFaces here
            for facade in footprint.facades:
                r.createFace(building, facade.indices)