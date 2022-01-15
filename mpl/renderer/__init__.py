import parse
from parse.osm import Osm
from .. import Mpl


class Renderer:
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def renderString(self, v1, v2, string):
        if string:
            self.mpl.ax.text(
                (v1[0]+v2[0])/2.,
                (v1[1]+v2[1])/2.,
                ' '+str(string)
            )
    
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


class BuildingBaseRenderer(Renderer):

    def render(self, building, data):
        element = building.element
        # render the outer footprint
        self.renderBuildingFootprint(building)
        # render holes for a multipolygon
        if element.t is parse.multipolygon:
            for l in element.ls:
                if not l.role is Osm.outer:
                    self.renderLineString(element.getLinestringData(l, data), True, **BuildingRenderer.style)

    def renderBuildingFootprint(self, building):
        ax = self.mpl.ax
        
        ax.fill(
            tuple(vector.v1[0] for vector in building.polygon.vectors),
            tuple(vector.v1[1] for vector in building.polygon.vectors),
            '#d9d0c9'
        )
        for vector in building.polygon.vectors:
            if vector.skip:
                continue
            v1, v2 = vector.v1, vector.v2
            ax.plot(
                (v1[0], v2[0]),
                (v1[1], v2[1]),
                linewidth = 1.,
                color = '#c2b5aa'
            )


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