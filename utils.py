import bmesh

def extrudeMesh(bm, thickness):
	"""
	Extrude bmesh
	"""
	geom = bmesh.ops.extrude_face_region(bm, geom=bm.faces)
	verts_extruded = [v for v in geom["geom"] if isinstance(v, bmesh.types.BMVert)]
	bmesh.ops.translate(bm, verts=verts_extruded, vec=(0, 0, thickness))