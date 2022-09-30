from renderer import Renderer
from renderer.curve_renderer import CurveRenderer

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh

from mathutils import Vector


class StreetRenderer:
    
    def __init__(self):
        self.streetSectionsCollection = None
    
    def prepare(self):
        self.streetSectionsCollection = createCollection("Street sections", Renderer.collection)
        self.intersectionAreasObj = createMeshObject(
            "Intersections",
            collection = Renderer.collection
        )
    
    def render(self, manager, data):
        # get action <action.generate_streets.StreetGenerator>
        from action.generate_streets import StreetGenerator
        
        for streetGenerator in manager.actions:
            if isinstance(streetGenerator, StreetGenerator):
                break
        else:
            # no action <action.generate_streets.StreetGenerator> was found
            return
        
        self.renderStreetSections(streetGenerator)
        self.renderIntersections(streetGenerator)
    
    def renderStreetSections(self, streetGenerator):
        location = Vector((0., 0., 0.))
        for streetSection in streetGenerator.waySectionLines.values():
            centerline = streetSection.centerline
            obj = CurveRenderer.createBlenderObject(
                streetSection.tags.get("name", "street section"),
                location,
                self.streetSectionsCollection,
                None
            )
            spline = obj.data.splines.new('POLY')
            spline.points.add(len(centerline)-1)
            for index,point in enumerate(centerline):
                spline.points[index].co = (point[0], point[1], 0., 1.)

    def renderIntersections(self, streetGenerator):
        bm = getBmesh(self.intersectionAreasObj)
        
        for intersectionArea in streetGenerator.intersectionAreas:
            polygon = intersectionArea.polygon
            bm.faces.new(
                bm.verts.new(Vector((vert[0], vert[1], 0.))) for vert in polygon
            )
        
        setBmesh(self.intersectionAreasObj, bm)        
    
    def finalize(self):
        return
    
    def cleanup(self):
        return