"""Resolution-only Blender adapter; actual ion tile importing is deferred."""
from time import sleep

from .cesium_ion import resolve_asset
from ..downloads.transport import DownloadTransport
from ..progressive.core import Job
from ..progressive.blender import notify_result


RESOLUTION_MESSAGE = ('Cesium ion access resolved successfully. '
                      'Tile import through Cesium ion is not implemented yet.')


class IonAccessResolution:
    revision = 0  # Resolution does not create or modify Blender scene data.

    def __init__(self, token, context, transport=None):
        self.scene = context.scene
        self.view_layer = context.view_layer
        self.job = Job(resolve_asset(token), transport if transport is not None else
                       DownloadTransport(max_requests=1, max_bytes=1024 * 1024))

    @property
    def status(self):
        return 'Resolving Google 3D Tiles access through Cesium ion'

    def cancel(self):
        self.job.cancel()

    def finish(self, silent=False):
        # Provider credentials are per-operation and are not persisted in Blender.
        self.job.result = None
        if silent:
            return
        level, message = self.outcome()
        notify_result(level, message)

    def outcome(self):
        if self.job.cancelled:
            return 'INFO', 'Cesium ion access resolution cancelled'
        if self.job.error is not None:
            return 'ERROR', str(self.job.error)
        return 'INFO', RESOLUTION_MESSAGE

    def run_blocking(self):
        try:
            while not self.job.done:
                self.job.step()
                if self.job.waiting:
                    sleep(.001)
            return self.outcome()
        finally:
            if not self.job.done:
                self.job.cancel()
            self.job.result = None
