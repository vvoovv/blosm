

class Facade:
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
        self.bottomRenderer = itemRenderers["Bottom"]
    
    def render(self, footprint):
        # <r> is the global building renderer
        r = self.r
        
        facadeStyle = footprint.facadeStyle
        if footprint.facadeStyle:
            for facade in footprint.facades:
                if facade.visible:
                    for styleBlock in facadeStyle:
                        # Temporarily set <facade.styleBlock> to <_tyleBlock> to use attributes
                        # from <styleBlock> in the condition evaluation
                        facade.styleBlock = styleBlock
                        if facade.evaluateCondition(styleBlock):
                            # check if <minWidth> attribute is given in the styleBlock
                            minWidth = facade.getStyleBlockAttr("minWidth")
                            if (minWidth and facade.width > minWidth) or not minWidth:
                                if facade.getStyleBlockAttr("withoutRepeat") and facade.getStyleBlockAttrDeep("cl"):
                                    self.renderWithoutRepeat(facade)
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
                                        r.createFace(footprint, facade.indices),
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
                            r.createFace(footprint, facade.indices),
                            facade.uvs
                        )
        else:
            # simply create BMFaces here
            for facade in footprint.facades:
                r.createFace(footprint, facade.indices)