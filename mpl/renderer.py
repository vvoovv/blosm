import parse
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
        if building.outline.t is parse.polygon:
            self.renderBuildingFootprint(building)
        else:
            # multipolygon
            for coords in building.outline.getDataMulti(data):
                pass

    def renderBuildingFootprint(self, building):
        polygon = building.polygon
        allVerts = polygon.allVerts
        indices = polygon.indices
        
        for edgeIndex in range(polygon.n-1):
            vert1 = allVerts[indices[edgeIndex]]
            vert2 = allVerts[indices[edgeIndex+1]]
            self.mpl.ax.plot(
                (vert1[0], vert2[0]),
                (vert1[1], vert2[1]),
                linewidth = 1.,
                color = BuildingVisibilityRender.getFootprintEdgeColor(building, edgeIndex)
            )
        
        vert1 = allVerts[indices[-1]]
        vert2 = allVerts[indices[0]]
        self.mpl.ax.plot(
            (vert1[0], vert2[0]),
            (vert1[1], vert2[1]),
            linewidth = 1.,
            color = BuildingVisibilityRender.getFootprintEdgeColor(building, -1)
        )
    
    @staticmethod
    def getFootprintEdgeColor(building, edgeIndex):
        visibility = building.visibility[0][building.polygon.indices[edgeIndex]]
        return 'red' if not visibility else (
            'green' if visibility > 0.75 else (
                'yellow' if visibility > 0.3 else 'blue'
            )
        )