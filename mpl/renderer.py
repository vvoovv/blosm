import parse
from parse.osm import Osm
from way.manager import facadeVisibilityWayCategories
from . import Mpl


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
        linewidth = 1.,
        color = "brown"
    )
    
    def render(self, way, data):
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


class BuildingVisibilityRender(Renderer):
    
    def render(self, building, data):
        outline = building.outline
        # render the outer footprint
        self.renderBuildingFootprint(building)
        # render holes for a multipolygon
        if outline.t is parse.multipolygon:
            for l in outline.ls:
                if not l.role is Osm.outer:
                    self.renderLineString(outline.getLinestringData(l, data), True, BuildingRenderer.style)

    def renderBuildingFootprint(self, building):
        for edge in building.polygon.edges:
            self.mpl.ax.plot(
                (edge.v1[0], edge.v2[0]),
                (edge.v1[1], edge.v2[1]),
                linewidth = 1.,
                color = BuildingVisibilityRender.getFootprintEdgeColor(edge)
            )
    
    @staticmethod
    def getFootprintEdgeColor(edge):
        visibility = edge.visibility
        return 'red' if not visibility else (
            'green' if visibility > 0.75 else (
                'yellow' if visibility > 0.3 else 'blue'
            )
        )


class WayVisibilityRenderer(WayRenderer):
    
    def render(self, way, data):
        if way.category in facadeVisibilityWayCategories:
            super().render(way, data)