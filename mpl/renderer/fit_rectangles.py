from . import Renderer
from fit_rectangles.fit_rectangles import fit_rectangles


class FitRectanglesRenderer(Renderer):
    
    def render(self, building, data):
        polygon = building.polygon
        verts = tuple(polygon.verts)
        
        rectangles = fit_rectangles(
            verts, 0, polygon.numEdges
        )
        
        self.renderLineString(verts, True, linewidth=1, color="black")
        
        for rectangle in rectangles:
            self.renderLineString(rectangle, True, linewidth=1, color="blue")