def assignTags(obj, tags):
    for key in tags:
        obj[key] = tags[key]

def parse_scalar_and_unit( htag ):
    for i,c in enumerate(htag):
        if not c.isdigit():
            return int(htag[:i]), htag[i:].strip()

    return int(htag), ""

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

