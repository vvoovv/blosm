import parse
from .. import Mpl


class Renderer:
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def renderLineString(self, coords, closed, style):
        prevCoord = coord0 = None
        for coord in coords:
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
        linewidth = 2.,
        color = "brown"
    )
    
    def render(self, way, data):
        # st = WayRenderer.style
        # level = WayLevel[way.category]
        # st['linewidth'] = 4 - level
        # self.renderLineString(way.element.getData(data), way.element.isClosed(), st)
        self.renderLineString(way.element.getData(data), way.element.isClosed(), WayRenderer.style)


class BuildingRenderer(Renderer):
    
    style = dict(
        linewidth = 1.,
        color = "gray"
    )
    
    def render(self, building, data):
        if building.outline.t is parse.polygon:
            self.renderLineString(building.outline.getData(data), True, BuildingRenderer.style)
        else:
            # multipolygon
            for coords in building.outline.getDataMulti(data):
                self.renderLineString(coords, True, BuildingRenderer.style)