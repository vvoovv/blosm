

class Facade:
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
        self.bottomRenderer = itemRenderers["Bottom"]
    
    def render(self, footprint, data):
        # <r> is the global building renderer
        r = self.r
        building = footprint.building
        
        if footprint.entranceAttr:
            footprint.processFacades(data)
        
        facadeStyle = footprint.facadeStyle
        if footprint.facadeStyle:
            for facade in footprint.facades:
                for styleBlock in facadeStyle:
                    if facade.evaluateCondition(styleBlock):
                        facade.styleBlock = styleBlock
                        # check if <minWidth> attribute is given in the styleBlock
                        minWidth = facade.getStyleBlockAttr("minWidth")
                        if (minWidth and facade.width > minWidth) or not minWidth:
                            facadeClass = facade.getStyleBlockAttr("cl")
                            if facadeClass:
                                self.renderClass(
                                    facade,
                                    facadeClass,
                                    r.createFace(building, facade.indices),
                                    None
                                )
                                break
                            elif styleBlock.markup:
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
                                # No markup, so we render cladding only.
                                self.renderCladding(
                                    facade,
                                    r.createFace(building, facade.indices),
                                    facade.uvs
                                )
                                break
                        # Clean up the styleBlock for the next attempt with
                        # the next style block from <facadeStyle>
                        facade.styleBlock = None
                else:
                    # No style block suits the <facade>
                    # Use style of <footprint> to render cladding for <facade>
                    self.renderCladding(
                        footprint,
                        r.createFace(building, facade.indices),
                        facade.uvs
                    )
        else:
            # simply create BMFaces here
            for facade in footprint.facades:
                r.createFace(building, facade.indices)