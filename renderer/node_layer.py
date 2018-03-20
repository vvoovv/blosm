from .layer import Layer
from util import zeroVector


class NodeLayer(Layer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        self.parentLocation = zeroVector()