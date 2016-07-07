import bpy, bmesh

def extrudeMesh(bm, thickness, face=None):
    """
    Extrude bmesh
    """
    geom = bmesh.ops.extrude_face_region(bm, geom=(face,) if face else bm.faces)
    verts_extruded = [v for v in geom["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=verts_extruded, vec=(0, 0, thickness))


def assignMaterials(obj, materialname, color, faces):
    # Get material
    if bpy.data.materials.get(materialname) is not None:
        mat = bpy.data.materials[materialname]
    else:
        # create material
        mat = bpy.data.materials.new(name=materialname)
        mat.diffuse_color = color

    # Assign it to object
    matidx = len(obj.data.materials)
    obj.data.materials.append(mat) 

    for face in faces:
        face.material_index = matidx