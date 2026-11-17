"""3D Tiles adapter for the shared progressive import driver."""
import bpy
from .blender import BlenderRenderer
from ..progressive.core import Job
from ..downloads.transport import DownloadTransport
from ..progressive.blender import preserve_context, notify_result as notify


class ProgressiveRenderer(BlenderRenderer):
    def prepare(self, manager):
        self.num_imported_tiles = 0
        self.collection = None
        self.cancelled = False
        self.revision = 0

    def renderGlb(self, manager, uri, path, transformMatrix, *, content=None):
        self._tile(manager, uri, path, transformMatrix, content, 'renderGlb')

    def renderB3dm(self, manager, uri, path, transformMatrix, *, content=None):
        self._tile(manager, uri, path, transformMatrix, content, 'renderB3dm')

    def _destination(self):
        # Users can delete or unlink the collection between timer turns.
        # Follow its identity, not its name, including after renaming or moving
        # it within the scene. Never adopt an unrelated same-named collection.
        collection = self.collection
        if collection is not None:
            try:
                collection.name  # A deleted Blender datablock raises ReferenceError.
                if collection in bpy.context.scene.collection.children_recursive:
                    return collection
            except ReferenceError:
                pass
            raise RuntimeError('The destination collection was deleted or removed from the import scene')
        return None

    def _tile(self, manager, uri, path, transform, content, method):
        destination = self._destination()
        with preserve_context():
            tile = BlenderRenderer(self.threedTilesName, False, self.instanceName,
                                   report=self.report, recordAttribution=False)
            tile.heightOffset = self.heightOffset
            tile.calculateHeightOffset = self.calculateHeightOffset
            output = None
            existing_objects = set(bpy.data.objects)
            existing_collections = set(bpy.data.collections)
            new_data = [(getattr(bpy.data, name), set(getattr(bpy.data, name))) for name in
                        ('meshes', 'curves', 'armatures', 'cameras', 'lights', 'materials', 'images', 'actions')]
            try:
                tile.prepare(manager)
                output = tile.collection
                getattr(tile, method)(manager, uri, path, transform, content=content)
                tile.finalize(manager)
                if not tile.num_imported_tiles:
                    return
                if destination is None:
                    self.collection = output
                else:
                    for obj in list(output.objects):
                        destination.objects.link(obj)
                        output.objects.unlink(obj)
                    bpy.data.collections.remove(output)
                for holder, count in tile.copyrightHolders.items():
                    self.copyrightHolders[holder] = self.copyrightHolders.get(holder, 0) + count
                if self.calculateHeightOffset:
                    self.heightOffset = (min(self.heightOffset, tile.heightOffset)
                                         if self.num_imported_tiles else tile.heightOffset)
                self.num_imported_tiles += tile.num_imported_tiles
                self.revision += 1
                self._attribution()
            except Exception:
                # A failed glTF may have created partial objects. Remove only
                # objects/collections created in this synchronous tile step.
                for obj in set(bpy.data.objects) - existing_objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                for collection in set(bpy.data.collections) - existing_collections:
                    bpy.data.collections.remove(collection)
                for blocks, existing in new_data:
                    for block in set(blocks) - existing:
                        if block.users == 0:
                            blocks.remove(block)
                raise

    def finalize(self, manager):
        # Completed tiles are already positioned. Cancel never joins or
        # retransforms the user's visible objects.
        if self.cancelled:
            return
        collection = self._destination()
        if self.join3dTilesObjects and collection is not None:
            with preserve_context():
                bpy.ops.object.select_all(action='DESELECT')
                for obj in collection.objects:
                    if obj.type == 'MESH':
                        obj.select_set(True)
                meshes = list(bpy.context.selected_objects)
                if meshes:
                    # Bake translations before joining so Blender does not
                    # subtract and re-add a distant active object's origin.
                    bpy.context.scene.cursor.location = (0., 0., 0.)
                    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                    bpy.context.view_layer.objects.active = meshes[0]
                    if len(meshes) > 1:
                        bpy.ops.object.join()
                    meshes[0].name = self.threedTilesName
                    self.revision += 1


class TilesImport:
    def __init__(self, manager, bounds, context, set_origin, transport=None):
        self.manager = manager
        self.renderer = manager.renderer
        self.scene = context.scene
        self.view_layer = context.view_layer
        self.set_origin = set_origin
        self.renderer.num_imported_tiles = 0
        self.renderer.revision = 0
        manager.errors = []
        self.job = Job(manager.iterRender(*bounds), transport or DownloadTransport(
                           max_requests=manager.downloadWorkers if manager.concurrentDownloads else 1),
                       validate=self.validate)

    def validate(self):
        # Check before resuming downloads/traversal so a removed destination
        # stops the job, rather than being counted as a failure for each tile.
        if self.renderer.num_imported_tiles:
            self.renderer._destination()

    @property
    def revision(self):
        # Update origin as soon as the first tile is visible, including on cancel.
        if self.renderer.num_imported_tiles:
            if self.set_origin:
                self.scene['lat'], self.scene['lon'] = self.manager.centerLat, self.manager.centerLon
            if self.renderer.calculateHeightOffset:
                self.scene['height_offset'] = self.renderer.heightOffset
        return self.renderer.revision

    @property
    def status(self):
        if getattr(self.manager, 'resolvingIon', False):
            return 'Resolving Google 3D Tiles access through Cesium ion'
        return '%s tiles imported, %s errors%s' % (
            self.renderer.num_imported_tiles, len(self.manager.errors),
            ' — downloading' if self.job.waiting else '')

    def cancel(self):
        self.renderer.cancelled = True
        self.job.cancel()

    def finish(self, silent=False):
        try:
            self.revision  # Persist origin/height for tiles retained on cancellation.
        except ReferenceError:
            pass  # The owning scene may have been deleted by the user.
        for error in self.manager.errors:
            print(error)
        if silent:
            return
        count = self.renderer.num_imported_tiles
        errors = len(self.manager.errors)
        if self.job.error:
            notify('ERROR', '3D Tiles import stopped: %s' % self.job.error)
        elif self.job.cancelled:
            notify('INFO', 'Import cancelled. Kept %s 3D Tiles; %s errors. See the System Console.' % (count, errors))
        elif self.job.result and len(self.job.result) == 1:
            notify('ERROR', self.job.result[0])
        elif not count:
            notify('ERROR', 'No 3D tiles were imported. %s errors occurred during the import. See the System Console.' % errors)
        elif errors:
            notify('INFO',
                   'Imported %s 3D Tiles. %s errors occurred during the import. See the System Console.' % (count, errors))
        else:
            notify('INFO', 'Successfully imported %s 3D Tiles.' % count)
