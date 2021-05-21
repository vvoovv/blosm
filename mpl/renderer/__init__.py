import parse
from .. import Mpl


class Renderer:
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def renderId(self, v1, v2, id):
        self.mpl.ax.text((v1[0]+v2[0])/2., (v1[1]+v2[1])/2., ' '+str(id) )
    
    def renderLineString(self, coords, closed, **style):
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
    
    def render(self, way, data):
        # st = WayRenderer.style
        # level = WayLevel[way.category]
        # st['linewidth'] = 4 - level
        # self.renderLineString(way.element.getData(data), way.element.isClosed(), st)
        for segment in way.segments:
            self.renderWaySegment(segment)
    
    def renderWaySegment(self, segment):
        self.mpl.ax.plot(
            (segment.v1[0], segment.v2[0]),
            (segment.v1[1], segment.v2[1]),
            linewidth = self.getLineWidth(segment),
            color = self.getColor(segment)
        )
    
    def getLineWidth(self, waySegment):
        return 2.
    
    def getColor(self, waySegment):
        return "brown"


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