import parse
from . import Mpl


class Renderer:
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def renderLineString(self, element, data, style):
        closed = element.isClosed()
        prevCoord = coord0 = None
        for coord in element.getData(data):
            if prevCoord:
                self.mpl.ax.plot(
                    (prevCoord[0], coord[0]),
                    (prevCoord[1], coord[1]),
                    **style
                )
            elif closed:
                coord0 = coord
            prevCoord = coord
        if closed:
            self.mpl.ax.plot(
                (coord[0], coord0[0]),
                (coord[1], coord0[1]),
                **style
            )

    def prepare(self):
        pass
    
    def finalize(self):
        self.mpl.show()
    
    def cleanup(self):
        self.mpl = None
        Mpl.cleanup()


class WayRenderer(Renderer):
    """
    A renderer for physical ways
    """
    
    style = dict(
        linewidth = 1.,
        color = "brown"
    )
    
    def render(self, way, data):
        self.renderLineString(way.element, data, WayRenderer.style)


class BuildingRenderer(Renderer):
    
    style = dict(
        linewidth = 1.,
        color = "gray"
    )
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def render(self, building, data):
        if building.outline.t is parse.polygon:
            self.renderLineString(building.outline, data, BuildingRenderer.style)
        else:
            pass