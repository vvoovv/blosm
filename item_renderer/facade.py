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
                    if facade.evaluateCondition(styleBlock) and facade.checkWidth(styleBlock):
                        if facade.markup:
                            self.renderMarkup(facade)
                        else:
                            pass
        else:
            # simply create BMFaces here
            for facade in footprint.facades:
                r.createFace(building, facade.indices, facade.uvs)
                