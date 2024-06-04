import bpy
from mathutils import Vector, Quaternion, Matrix


def set_convert_functions_4_0(gltf):
    if bpy.app.debug_value != 100:
        # Unit conversion factor in (Blender units) per meter
        u = 1.0 / bpy.context.scene.unit_settings.scale_length
        
        offset = gltf._offset
        # We will apply <offset> before creating a <Vector> to decrease numerical errors caused by
        # 32 bits floats useb by <Vector> and other <mathutils> classes.

        # glTF Y-Up space --> Blender Z-up space
        # X,Y,Z --> X,-Z,Y
        def convert_loc(x): return u * Vector([x[0] - offset[0], -x[2] - offset[1], x[1] - offset[2]])
        def convert_quat(q): return Quaternion([q[3], q[0], -q[2], q[1]])
        def convert_scale(s): return Vector([s[0], s[2], s[1]])
        def convert_matrix(m):
            return Matrix([
                [   m[0],   -m[ 8],    m[4],  m[12]*u],
                [  -m[2],    m[10],   -m[6], -m[14]*u],
                [   m[1],   -m[ 9],    m[5],  m[13]*u],
                [ m[3]/u, -m[11]/u,  m[7]/u,    m[15]],
            ])

        # Batch versions operate in place on a numpy array
        def convert_locs_batch(locs):
            # x,y,z -> x,-z,y
            locs[:, [1,2]] = locs[:, [2,1]]
            locs[:, 1] *= -1
            # Unit conversion
            if u != 1: locs *= u
        def convert_normals_batch(ns):
            ns[:, [1,2]] = ns[:, [2,1]]
            ns[:, 1] *= -1

        # Correction for cameras and lights.
        # glTF: right = +X, forward = -Z, up = +Y
        # glTF after Yup2Zup: right = +X, forward = +Y, up = +Z
        # Blender: right = +X, forward = -Z, up = +Y
        # Need to carry Blender --> glTF after Yup2Zup
        gltf.camera_correction = Quaternion((2**0.5/2, 2**0.5/2, 0.0, 0.0))

    else:
        def convert_loc(x): return Vector(x)
        def convert_quat(q): return Quaternion([q[3], q[0], q[1], q[2]])
        def convert_scale(s): return Vector(s)
        def convert_matrix(m):
            return Matrix([m[0::4], m[1::4], m[2::4], m[3::4]])

        def convert_locs_batch(_locs): return
        def convert_normals_batch(_ns): return

        # Same convention, no correction needed.
        gltf.camera_correction = None

    gltf.loc_gltf_to_blender = convert_loc
    gltf.locs_batch_gltf_to_blender = convert_locs_batch
    gltf.quaternion_gltf_to_blender = convert_quat
    gltf.normals_batch_gltf_to_blender = convert_normals_batch
    gltf.scale_gltf_to_blender = convert_scale
    gltf.matrix_gltf_to_blender = convert_matrix


def select_imported_objects_4_1(gltf):
    return