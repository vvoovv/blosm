from . import Renderer, BuildingRenderer, WayRenderer
import parse
from parse.osm import Osm
from defs.way import facadeVisibilityWayCategoriesSet, Category
from defs.facade_classification import FacadeClass, WayLevel
from math import atan2, pi


class BuildingVisibilityRender(Renderer):
    
    def __init__(self, showAssoc):
        super().__init__()
        self.showAssoc = showAssoc

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

            if self.showAssoc:
                # visalization of association between way-segment and edge
                visInfo = edge.visInfo
                if hasattr(visInfo,'waySegment') and visInfo.value>=0.5:
                    seg = visInfo.waySegment
                    s1, s2 = seg.v1, seg.v2
                    ax.plot(s1[0], s1[1], 'k.', markersize=5.)
                    vx = (v1[0]+v2[0])/2.
                    vy = (v1[1]+v2[1])/2.
                    sx = (s1[0]+s2[0])/2.
                    sy = (s1[1]+s2[1])/2.
                    color = 'blue' if edge.cl==FacadeClass.deadend else (
                        'green' if edge.cl==FacadeClass.passage else 'magenta'
                        )
                    ax.annotate(
                        '',
                        xytext = (sx,sy),
                        xy=(vx,vy),
                        arrowprops=dict(color=color, width = 0.25, shrink=0., headwidth=3, headlength=8)
                    )
                    a = atan2(visInfo.dy,visInfo.dx)/pi*180.
                    ax.text((sx+vx)/2.,(sy+vy)/2.,' %4.2f,%2.0f°'%(visInfo.value, a))
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
    
    def __init__(self, sideFacadeColor, showAssoc):
        super().__init__()
        self.sideFacadeColor = sideFacadeColor
        self.showAssoc = showAssoc
    
    def render(self, building, data):
        outline = building.outline
        # render the outer footprint
        self.renderBuildingFootprint(building)
        # render holes for a multipolygon
        if outline.t is parse.multipolygon:
            for l in outline.ls:
                if not l.role is Osm.outer:
                    self.renderLineString(outline.getLinestringData(l, data), True, **BuildingRenderer.style)

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
            color = self.getFootprintEdgeColor(edge)
            linewidth = BuildingClassificationRender.getLineWidth(edge)
            ax.plot(
                (v1[0], v2[0]),
                (v1[1], v2[1]),
                linewidth = linewidth,
                color = color
            )
            ax.plot(v1[0], v1[1], 'k.', markersize=2.)

            if self.showAssoc:
                # visalization of association between way-segment and edge
                visInfo = edge.visInfo
                if hasattr(visInfo,'waySegment') and visInfo.value>=0.5:
                    seg = visInfo.waySegment
                    s1, s2 = seg.v1, seg.v2
                    ax.plot(s1[0], s1[1], 'k.', markersize=5.)
                    vx = (v1[0]+v2[0])/2.
                    vy = (v1[1]+v2[1])/2.
                    sx = (s1[0]+s2[0])/2.
                    sy = (s1[1]+s2[1])/2.
                    color = 'blue' if edge.cl==FacadeClass.deadend else (
                        'green' if edge.cl==FacadeClass.passage else 'magenta'
                        )
                    ax.annotate(
                        '',
                        xytext = (sx,sy),
                        xy=(vx,vy),
                        arrowprops=dict(color=color, width = 0.25, shrink=0., headwidth=3, headlength=8)
                    )
                    a = atan2(visInfo.dy,visInfo.dx)/pi*180.
                    ax.text((sx+vx)/2.,(sy+vy)/2.,' %4.2f,%2.0f°'%(visInfo.value, a))

    def getFootprintEdgeColor(self, edge):
        cl = edge.cl
        return 'gray' if cl == FacadeClass.unknown else (
            'green' if cl == FacadeClass.front else (
                self.sideFacadeColor if cl == FacadeClass.side else (
                    'red' if cl == FacadeClass.back else (
                        'magenta' if cl == FacadeClass.passage else (
                            'cyan' if cl == FacadeClass.deadend else 'black'
                        )
                    )
                )
            )
        )
    
    @staticmethod
    def getLineWidth(edge):
        return 0.5 if edge.cl == FacadeClass.shared else 1.5


class WayVisibilityRenderer(WayRenderer):
    
    def render(self, way, data):
        if way.category in facadeVisibilityWayCategoriesSet:
            super().render(way, data)
    
    def getLineWidth(self, way):
        return 0.5 if way.category == Category.service else super().getLineWidth(way)