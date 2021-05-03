import parse
from parse.osm import Osm
from way.manager import facadeVisibilityWayCategories
from action.facade_classification import FacadeClass
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
        linewidth = 2.,
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
        ax = self.mpl.ax
        
        ax.fill(
            tuple(vector.v1[0] for vector in building.polygon.vectors),
            tuple(vector.v1[1] for vector in building.polygon.vectors),
            '#f5f5dc'
        )
        for vector in building.polygon.vectors:
            if vector.skip:
                continue
            edge, v1, v2 = vector.edge, vector.v1, vector.v2
            color = BuildingVisibilityRender.getFootprintEdgeColor(edge)
            linewidth = BuildingVisibilityRender.getLineWidth(edge)
            ax.plot(
                (v1[0], v2[0]),
                (v1[1], v2[1]),
                linewidth = linewidth,
                color = color
            )
            ax.plot(v1[0], v1[1], 'k.', markersize=2.)
            #if not skip:
            #    ax.annotate(str(vector.index), xy=(v1[0], v1[1]))
            #if not edge.hasSharedBuildings():
            #    ax.annotate(
            #        '',
            #        xytext = (v1[0], v1[1]),
            #        xy=(v2[0], v2[1]),
            #        arrowprops=dict(color=color, width = 0.25, shrink=0., headwidth=3, headlength=8)
            #    )
    
    @staticmethod
    def getFootprintEdgeColor(edge):
        if edge.hasSharedBldgVectors():
            return 'black'
        visibility = edge.visInfo.value
        return 'red' if not visibility else (
            'green' if visibility > 0.75 else (
                'yellow' if visibility > 0.3 else 'blue'
            )
        )
    
    @staticmethod
    def getLineWidth(edge):
        return 0.5 if edge.hasSharedBldgVectors() else 1.5

class BuildingClassificationRender(Renderer):
    
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
        ax = self.mpl.ax
        
        ax.fill(
            tuple(vector.v1[0] for vector in building.polygon.vectors),
            tuple(vector.v1[1] for vector in building.polygon.vectors),
            '#f5f5dc'
        )
        for vector in building.polygon.vectors:
            if vector.skip:
                continue
            edge, v1, v2 = vector.edge, vector.v1, vector.v2
            color = BuildingClassificationRender.getFootprintEdgeColor(edge)
            linewidth = BuildingClassificationRender.getLineWidth(edge)
            ax.plot(
                (v1[0], v2[0]),
                (v1[1], v2[1]),
                linewidth = linewidth,
                color = color
            )
            ax.plot(v1[0], v1[1], 'k.', markersize=2.)
    
    @staticmethod
    def getFootprintEdgeColor(edge):
        cl = edge.cl
        return 'gray' if cl == FacadeClass.unknown else (
            'green' if cl == FacadeClass.front else (
                'yellow' if cl == FacadeClass.side else (
                    'red' if cl == FacadeClass.back else (
                        'magenta' if cl == FacadeClass.passage else 'black'
                    )
                )
            )
        )
    
    @staticmethod
    def getLineWidth(edge):
        return 0.5 if edge.cl == FacadeClass.shared else 1.5

class WayVisibilityRenderer(WayRenderer):
    
    def render(self, way, data):
        if way.category in facadeVisibilityWayCategories:
            super().render(way, data)


#########################################################
# The classes for debugging profiled roofs
#########################################################

class RoofProfileRenderer(Renderer):
    
    def __init__(self, app, data):
        from action.volume import Volume
        from building2.item_store import ItemStore
        from building2.item_factory import ItemFactory
        from item.footprint import Footprint
        from item.facade import Facade
        from item.roof_profile import RoofProfile
        from item.roof_side import RoofSide
        
        super().__init__()
        
        self.itemStore = ItemStore( (Footprint(''),) )
        
        self.itemFactory = ItemFactory( (Footprint(''), RoofProfile(), RoofSide(), Facade()) )
        
        self.volumeAction = Volume(
            app,
            data,
            self.itemStore,
            self.itemFactory,
            dict(
                Facade = FacadeItemRenderer(),
                RoofProfile = RoofProfileItemRenderer(self)
            )
        )
    
    def render(self, buildingP, data):
        from item.footprint import Footprint
        from grammar.value import FromAttr
        from item.defs import RoofShapes
        class DummyBuilding:
            def __init__(self):
                self.verts = []
                self.altitudeDifference = 0.
        class DummyFootprintStyle:
            def __init__(self):
                self.styleBlocks = dict(
                    Footprint=tuple()
                )
                self.condition = None
                self.attrs = dict(
                    roofShape = (FromAttr("roof:shape", FromAttr.String, RoofShapes), True),
                    numLevels = (FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive), True)
                )
        class DummyBuildingStyle:
            def __init__(self):
                self.styleBlocks = dict(
                    Footprint = (DummyFootprintStyle(),)
                )
        
        footprint = Footprint('')
        footprint.element = buildingP.outline
        building = DummyBuilding()
        footprint.building = building
        
        self.itemStore.add(footprint)
        
        self.volumeAction.do(building, Footprint, DummyBuildingStyle(), self)
        
        self.itemStore.clear()


class FacadeItemRenderer():
    
    def render(self, footprint, data):
        pass


class RoofProfileItemRenderer():
    
    def __init__(self, renderer):
        # a global renderer
        self.renderer = renderer
    
    def render(self, roofItem):
        from util import zAxis
        
        footprint = roofItem.footprint
        
        ax = self.renderer.mpl.ax
        roofProfileVolumeGenerator = self.renderer.volumeAction.volumeGenerators[footprint.getStyleBlockAttr("roofShape")]
        verts = roofItem.building.verts
        polygon = roofItem.footprint.polygon
        polygonWidth = footprint.polygonWidth
        direction = footprint.direction
        # a unit vector perpendicular to <direction>
        pDirection = zAxis.cross(direction)
        
        self.renderer.renderLineString(polygon.verts, True, BuildingRenderer.style)
        
        # The following vertices mark min X, min Y and max Y in the system of reference
        # formed by two perpendicular unit vectors <direction> and <pDirection>
        vertMinX = verts[footprint.minProjIndex]
        vertMinY = min((verts[i] for i in range(polygon.n)), key = lambda v: v.dot(pDirection))
        vertMaxY = max((verts[i] for i in range(polygon.n)), key = lambda v: v.dot(pDirection))
        polygonHeight = (vertMaxY - vertMinY).dot(pDirection)
        # <vert1> is the leftmost bottommost point of the bounding box with the axes
        # parallel to <direction> and <pDirection>
        vert1 = vertMinX.dot(direction)*direction + vertMinY.dot(pDirection)*pDirection
        vert2 = vert1 + polygonWidth*footprint.direction
        ax.plot(
            ( (vert1[0], vert2[0])  ),
            ( (vert1[1], vert2[1])  )
        )
        
        for x,_ in roofProfileVolumeGenerator.profile:
            _vert1 = vert1 + x * polygonWidth * direction
            _vert2 = _vert1 + polygonHeight*pDirection
            ax.plot(
                ( (_vert1[0], _vert2[0])  ),
                ( (_vert1[1], _vert2[1])  ),
                linewidth = 1.,
                color = 'black'
            )
        
        for vertIndex in range(polygon.n, len(verts)):
            vert = verts[vertIndex]
            ax.plot(vert[0], vert[1], 'k.', markersize=3.)
            ax.annotate(str(vertIndex), xy=(vert[0], verts[vertIndex][1]))