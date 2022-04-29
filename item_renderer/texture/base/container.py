from .item_renderer import ItemRendererMixin
from ..container import Container as ContainerBase


class Container(ContainerBase, ItemRendererMixin):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    pass