"""Private Blender process: evaluate one saved compositor scene, then exit.

No addon registration or automatic scripts. Inputs are copied by the parent.
"""
import sys
from pathlib import Path
import bpy


def main():
    snapshot, output = sys.argv[sys.argv.index('--')+1:]
    with bpy.data.libraries.load(snapshot) as (source, target):
        target.scenes = list(source.scenes)
    scene = target.scenes[0]
    nodes = scene.compositing_node_group.nodes if hasattr(scene, 'compositing_node_group') else scene.node_tree.nodes
    node = nodes['File Output']
    if hasattr(node, 'directory'):
        node.directory = output
    else:
        node.base_path = output
    result = bpy.ops.render.render(scene=scene.name)
    if result != {'FINISHED'}:
        raise RuntimeError('Texture rendering did not finish for scene "%s" in "%s"' % (scene.name, output))
    files = list(Path(output).iterdir())
    if len(files) != 1 or not files[0].is_file() or not files[0].stat().st_size:
        raise RuntimeError('Compositor must produce exactly one nonempty texture for scene "%s" in "%s"' % (scene.name, output))


if __name__ == '__main__':
    main()
