import bpy, bmesh
import utils, osm_utils


def getNumber(s):
    try:
        n = float(s)
    except ValueError:
        n = None
    return n


class Buildings:
    @staticmethod
    def condition(tags, way):
        return "building" in tags
    
    @staticmethod
    def handler(way, parser, kwargs):
        singleMesh = kwargs["bm"]
        
        wayNodes = way["nodes"]
        numNodes = len(wayNodes)-1 # we need to skip the last node which is the same as the first ones
        # a polygon must have at least 3 vertices
        if numNodes<3:
            return
        
        if not singleMesh:
            tags = way["tags"]
            osmId = way["id"]
            # compose object name
            name = osmId
            if "addr:housenumber" in tags and "addr:street" in tags:
                name = tags["addr:street"] + ", " + tags["addr:housenumber"]
            elif "name" in tags:
                name = tags["name"]
        
        bm = kwargs["bm"] if singleMesh else bmesh.new()
        verts = []
        for node in range(numNodes):
            node = parser.nodes.get(wayNodes[node])
            if node is None:
                # skip that OSM way
                return
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], 0)) )
        
        face = bm.faces.new(verts)
        
        tags = way["tags"]
        height = kwargs["defaultHeight"]
        if "height" in tags:
            # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
            height = osm_utils.parse_scalar_and_unit(tags["height"])[0]
        elif "building:levels" in tags:
            numLevels = getNumber(tags["building:levels"])
            if not numLevels is None:
                height = numLevels * kwargs["levelHeight"]

        # extrude
        if height > 0.:
            utils.extrudeMesh(bm, height, face if singleMesh else None)
            
        if not singleMesh:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
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
        singleMesh = kwargs["bm"]
        
        wayNodes = way["nodes"]
        numNodes = len(wayNodes)-1 # we need to skip the last node which is the same as the first ones
        # a polygon must have at least 3 vertices
        if numNodes<3: return
        
        tags = way["tags"]
        if not singleMesh:
            osmId = way["id"]
            # compose object name
            name = osmId
            if "addr:housenumber" in tags and "addr:street" in tags:
                name = tags["addr:street"] + ", " + tags["addr:housenumber"]
            elif "name" in tags:
                name = tags["name"]

        min_height = 0.
        if "min_height" in tags:
            # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
            min_height = osm_utils.parse_scalar_and_unit(tags["min_height"])[0]
        elif "building:min_level" in tags:
            min_level = getNumber(tags["building:min_level"])
            if not min_level is None:
                min_height = min_level * kwargs["levelHeight"]
        
        height = kwargs["defaultHeight"]
        if "height" in tags:
            # There's a height tag. It's parsed as text and could look like: 25, 25m, 25 ft, etc.
            height = osm_utils.parse_scalar_and_unit(tags["height"])[0]
        elif "building:levels" in tags:
            numLevels = getNumber(tags["building:levels"])
            if not numLevels is None:
                height = numLevels * kwargs["levelHeight"]

        bm = kwargs["bm"] if singleMesh else bmesh.new()
        verts = []
        for node in range(numNodes):
            node = parser.nodes.get(wayNodes[node])
            if node is None:
                # skip that OSM way
                return
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], min_height)) )
        
        face = bm.faces.new(verts)
        
        # extrude
        if (height-min_height) > 0.:
            utils.extrudeMesh(bm, (height-min_height), face if singleMesh else None)
            
        if not singleMesh:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            
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
            node = parser.nodes.get(wayNodes[node])
            if node is None:
                # skip that OSM way
                return
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
            node = parser.nodes.get(wayNodes[node])
            if node is None:
                # skip that OSM way
                return
            v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
            verts.append( bm.verts.new((v[0], v[1], 0)) )
        
        bm.faces.new(verts)
        
        if not kwargs["bm"]:
            tags = way["tags"]
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            
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
