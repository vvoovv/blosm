import bpy, bmesh
import utils, osm_utils

class Buildings:
    @staticmethod
    def condition(tags, way):
        return "building" in tags
    
    @staticmethod
    def handler(way, parser, kwargs):        
        heightPerLevel = 2.70 # default height per level if building height is not specified
        wayNodes = way["nodes"]
        numNodes = len(wayNodes)-1 # we need to skip the last node which is the same as the first ones
        # a polygon must have at least 3 vertices
        if numNodes<3: return
        
        if not kwargs["bm"]: # not a single mesh
            tags = way["tags"]
            osmId = way["id"]
            # compose object name
            name = osmId
            if "addr:housenumber" in tags and "addr:street" in tags:
                name = tags["addr:street"] + ", " + tags["addr:housenumber"]
            elif "name" in tags:
                name = tags["name"]
        
        bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
        verts = []
        for node in range(numNodes):
            node = parser.nodes[wayNodes[node]]
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], 0)) )
        
        bm.faces.new(verts)
        
        if not kwargs["bm"]:
            tags = way["tags"]
            thickness = 0
            if "height" in tags:
                # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
                thickness,unit = osm_utils.parse_scalar_and_unit(tags["height"])
            elif "building:levels" in tags:
                # If no height is given, calculate height of Building from Levels
                thickness = int(tags["building:levels"])*heightPerLevel
            else:
                thickness = kwargs["thickness"] if ("thickness" in kwargs) else 0

            # extrude
            if thickness>0:
                utils.extrudeMesh(bm, thickness)
            
            bm.normal_update()
            
            mesh = bpy.data.meshes.new(osmId)
            bm.to_mesh(mesh)
            
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.scene.objects.link(obj)
            bpy.context.scene.update()
            
            # final adjustments
            obj.select = True
            # assign OSM tags to the blender object
            osm_utils.assignTags(obj, tags)

            utils.assignMaterials( obj, "roof", (1.0,0.0,0.0), [mesh.polygons[0]] )
            utils.assignMaterials( obj, "wall", (1,0.7,0.0), mesh.polygons[1:] )


class BuildingParts:
    @staticmethod
    def condition(tags, way):
        return "building:part" in tags
    
    @staticmethod
    def handler(way, parser, kwargs):
        heightPerLevel = 2.70 # default height per level if building height is not specified
        wayNodes = way["nodes"]
        numNodes = len(wayNodes)-1 # we need to skip the last node which is the same as the first ones
        # a polygon must have at least 3 vertices
        if numNodes<3: return
        
        tags = way["tags"]
        if not kwargs["bm"]: # not a single mesh
            osmId = way["id"]
            # compose object name
            name = osmId
            if "addr:housenumber" in tags and "addr:street" in tags:
                name = tags["addr:street"] + ", " + tags["addr:housenumber"]
            elif "name" in tags:
                name = tags["name"]

        min_height = 0
        height = 0
        if "min_height" in tags:
            # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
            min_height,unit = osm_utils.parse_scalar_and_unit(tags["min_height"])
        elif "building:min_level" in tags:
            # If no height is given, calculate height of Building from Levels
            min_height,unit = int(tags["building:min_level"])*heightPerLevel,""
            
        if "height" in tags:
            # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
            height,unit = osm_utils.parse_scalar_and_unit(tags["height"])
        elif "building:levels" in tags:
            # If no height is given, calculate height of Building from Levels
            height,unit = int(tags["building:levels"])*heightPerLevel,""

        bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
        verts = []
        for node in range(numNodes):
            node = parser.nodes[wayNodes[node]]
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], min_height)) )
        
        bm.faces.new(verts)
        
        if not kwargs["bm"]:
            tags = way["tags"]

            # extrude
            if (height-min_height)>0:
                utils.extrudeMesh(bm, (height-min_height))
            
            bm.normal_update()
            
            mesh = bpy.data.meshes.new(osmId)
            bm.to_mesh(mesh)
            
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.scene.objects.link(obj)
            bpy.context.scene.update()
            
            # final adjustments
            obj.select = True
            # assign OSM tags to the blender object
            osm_utils.assignTags(obj, tags)

class Highways:
    @staticmethod
    def condition(tags, way):
        return "highway" in tags
    
    @staticmethod
    def handler(way, parser, kwargs):
        wayNodes = way["nodes"]
        numNodes = len(wayNodes) # we need to skip the last node which is the same as the first ones
        # a way must have at least 2 vertices
        if numNodes<2: return
        
        if not kwargs["bm"]: # not a single mesh
            tags = way["tags"]
            osmId = way["id"]
            # compose object name
            name = tags["name"] if "name" in tags else osmId
        
        bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
        prevVertex = None
        for node in range(numNodes):
            node = parser.nodes[wayNodes[node]]
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            v = bm.verts.new((v[0], v[1], 0))
            if prevVertex:
                bm.edges.new([prevVertex, v])
            prevVertex = v
        
        if not kwargs["bm"]:
            mesh = bpy.data.meshes.new(osmId)
            bm.to_mesh(mesh)
            
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.scene.objects.link(obj)
            bpy.context.scene.update()
            
            # final adjustments
            obj.select = True
            # assign OSM tags to the blender object
            osm_utils.assignTags(obj, tags)
class Naturals:
    @staticmethod
    def condition(tags, way):
        return "natural" in tags
    
    @staticmethod
    def handler(way, parser, kwargs):
        wayNodes = way["nodes"]
        numNodes = len(wayNodes) # we need to skip the last node which is the same as the first ones
    
        if numNodes == 1:
            # This is some point "natural".
            # which we ignore for now (trees, etc.)
            pass

        numNodes = numNodes - 1

        # a polygon must have at least 3 vertices
        if numNodes<3: return
        
        tags = way["tags"]
        if not kwargs["bm"]: # not a single mesh
            osmId = way["id"]
            # compose object name
            name = osmId
            if "name" in tags:
                name = tags["name"]

        bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
        verts = []
        for node in range(numNodes):
            node = parser.nodes[wayNodes[node]]
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], 0)) )
        
        bm.faces.new(verts)
        
        if not kwargs["bm"]:
            tags = way["tags"]
            bm.normal_update()
            
            mesh = bpy.data.meshes.new(osmId)
            bm.to_mesh(mesh)
            
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.scene.objects.link(obj)
            bpy.context.scene.update()
            
            # final adjustments
            obj.select = True
            # assign OSM tags to the blender object
            osm_utils.assignTags(obj, tags)

            naturaltype = tags["natural"]
            color = (0.5,0.5,0.5)

            if naturaltype == "water":
                color = (0,0,1)

            utils.assignMaterials( obj, naturaltype, color, [mesh.polygons[0]] )
