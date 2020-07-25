import os
import bpy
from .item_renderer import ItemRendererMixin
from ..container import Container as ContainerBase

from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import setTextureSize, setTextureSize2


class Container(ContainerBase, ItemRendererMixin):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    pass