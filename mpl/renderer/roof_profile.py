from . import Renderer, BuildingRenderer


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
        
        self.renderer.renderLineString(polygon.verts, True, **BuildingRenderer.style)
        
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
        
        # slots
        for slotIndex,profile in enumerate(roofProfileVolumeGenerator.profile):
            x = profile[0]
            _vert1 = vert1 + x * polygonWidth * direction
            _vert2 = _vert1 + polygonHeight*pDirection
            ax.plot(
                ( (_vert1[0], _vert2[0])  ),
                ( (_vert1[1], _vert2[1])  ),
                linewidth = 1.,
                color = 'black'
            )
            ax.annotate(str(slotIndex), xy=(_vert1[0], _vert1[1]))
        
        for vertIndex in range(polygon.n, len(verts)):
            vert = verts[vertIndex]
            ax.plot(vert[0], vert[1], 'k.', markersize=3.)
            ax.annotate(str(vertIndex), xy=(vert[0], verts[vertIndex][1]))