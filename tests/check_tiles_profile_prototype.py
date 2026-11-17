"""Compare the diagnostic direct-finalize prototype to the unchanged importer.

Run in an isolated background Blender with --artifacts. This checks only the
static, unit-scale fixtures used by the profiling experiment, not arbitrary glTF.
"""
import argparse
import ast
from contextlib import nullcontext
import importlib
import json
from math import atan, sqrt, pi, radians
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gltf_test_support import bpy, PACKAGE, fixture, make_glb, reset_scene, fingerprint
import addon_utils

parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.artifacts.mkdir(parents=True,exist_ok=True)
addon_utils.enable(PACKAGE,default_set=False)
bpy.context.preferences.filepaths.temporary_directory=str(args.artifacts)
bpy.context.preferences.filepaths.file_preview_type='NONE'
render=importlib.import_module(PACKAGE+'.threed_tiles.blender')
tiles=importlib.import_module(PACKAGE+'.threed_tiles.progressive')
BaseManager=importlib.import_module(PACKAGE+'.threed_tiles.manager').BaseManager

# Execute only this experimental function, never the benchmark's timers/setup.
source=Path(__file__).with_name('benchmark_tiles_progressive.py')
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='direct_finalize')
scope=dict(bpy=bpy,atan=atan,sqrt=sqrt,pi=pi,radians=radians)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
prototype=scope['direct_finalize']
original=render.BlenderRenderer.finalize
center=BaseManager.fromGeographic(60.17,24.94,0)
doc,binary,_=fixture()
doc['nodes'][0]['translation']=[center.x,center.z,-center.y]
synthetic=make_glb(doc,binary)
samples=[('synthetic.glb',synthetic)]
directory=Path('C:/dev/workspace/artifacts/direct_gltf_import/helsinki/1')
samples.extend((path.name,path.read_bytes()) for path in sorted(directory.glob('mesh_*.b3dm')))
results=[]


def collect(name,content,finalize):
    reset_scene()
    render.BlenderRenderer.finalize=finalize
    renderer=tiles.ProgressiveRenderer('Parity',False,'')
    manager=BaseManager('https://parity.invalid/root.json',renderer)
    manager.centerLat,manager.centerLon=60.17,24.94
    renderer.prepare(manager)
    method=renderer.renderB3dm if name.endswith('.b3dm') else renderer.renderGlb
    legacy = (patch.object(render.BlenderRenderer,'_finalize_direct',return_value=False)
              if finalize is original and hasattr(render.BlenderRenderer,'_finalize_direct') else nullcontext())
    with legacy:
        method(manager,name,name,None,content=content)
        renderer.finalize(manager)
    assert renderer.num_imported_tiles==1
    bpy.context.view_layer.update()
    geometry=[[(tuple(obj.matrix_world @ vertex.co)) for vertex in obj.data.vertices]
              for obj in renderer.collection.objects if obj.type=='MESH']
    attributes=fingerprint()
    for entry in attributes:
        entry.pop('coordinates')
        entry.pop('matrix')
    return geometry,attributes


try:
    for name,content in samples:
        expected,expected_attributes=collect(name,content,original)
        actual,attributes=collect(name,content,prototype)
        assert len(actual)==len(expected)
        assert [len(mesh) for mesh in actual]==[len(mesh) for mesh in expected]
        error=max((abs(a-b) for mesh_a,mesh_b in zip(actual,expected)
                   for vertex_a,vertex_b in zip(mesh_a,mesh_b)
                   for a,b in zip(vertex_a,vertex_b)),default=0.)
        assert error<=.001,(name,error)
        assert attributes==expected_attributes,(name,'topology/UV/color/material mismatch')
        results.append(dict(sample=name,maximum_world_coordinate_error=error,
                            vertices=sum(map(len,actual))))
finally:
    render.BlenderRenderer.finalize=original
(args.artifacts/'prototype_parity.json').write_text(json.dumps(dict(status='passed',cases=results),indent=2))
print('PROTOTYPE_PARITY',json.dumps(results),flush=True)
