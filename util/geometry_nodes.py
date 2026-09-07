"""Version-independent access to Geometry Nodes modifier sockets.

Use socket identifiers, never modifier-key order or display names. Values are
per modifier; these helpers do not change a shared node tree's defaults.
See geometry_nodes.md for examples and the supported operations.
"""

import bpy


def _sockets(modifier, direction):
    if modifier.type != 'NODES':
        raise TypeError("Expected a Geometry Nodes modifier")
    group = modifier.node_group
    if group is None:
        raise ValueError("Geometry Nodes modifier has no node group")
    if hasattr(group, "interface"):
        return (
            item for item in group.interface.items_tree
            if item.item_type == 'SOCKET' and item.in_out == direction
        )
    return iter(group.inputs if direction == 'INPUT' else group.outputs)


def _socket(modifier, identifier, direction):
    for socket in _sockets(modifier, direction):
        if socket.identifier == identifier:
            return socket
    raise KeyError("Node group %r has no %s socket %r" % (
        modifier.node_group.name, direction.lower(), identifier
    ))


def _input(modifier, identifier):
    socket = _socket(modifier, identifier, 'INPUT')
    if bpy.app.version >= (5, 2, 0):
        value = getattr(modifier.properties.inputs, identifier)
        if hasattr(value, "value"):
            return value
    elif hasattr(socket, "default_value"):
        return None
    raise TypeError("Input %r does not support a modifier value" % identifier)


def iterGnInputs(modifier):
    """Yield value-bearing input sockets in interface order (skip geometry/panels)."""
    for socket in _sockets(modifier, 'INPUT'):
        if bpy.app.version >= (5, 2, 0):
            if hasattr(getattr(modifier.properties.inputs, socket.identifier), "value"):
                yield socket
        elif hasattr(socket, "default_value"):
            yield socket


def getGnInput(modifier, identifier):
    """Read the stored literal value, not an evaluated field or named attribute."""
    value = _input(modifier, identifier)
    return value.value if value is not None else modifier[identifier]


def setGnInput(modifier, identifier, value):
    """Set a literal socket value and switch off attribute/layer input mode."""
    prop = _input(modifier, identifier)
    if prop is not None:
        prop.value = value
        prop.type = 'VALUE'
    else:
        modifier[identifier] = value
        if identifier + "_use_attribute" in modifier:
            modifier[identifier + "_use_attribute"] = False
    modifier.id_data.update_tag()


def getGnInputAttribute(modifier, identifier):
    """Return the active named attribute, or None for a literal input.

    Blender's layer-selection mode is distinct and is not a named attribute.
    Reject it so callers copying inputs cannot silently change its meaning.
    """
    prop = _input(modifier, identifier)
    if prop is not None:
        if prop.type == 'ATTRIBUTE':
            return prop.attribute_name
        if prop.type != 'VALUE':
            raise ValueError("Input %r uses unsupported mode %r" % (identifier, prop.type))
    elif modifier.get(identifier + "_use_attribute", False):
        return modifier[identifier + "_attribute_name"]
    return None


def useAttributeForGnInput(modifier, identifier, attributeName):
    """Bind a field-capable input to a named attribute (an empty name is valid)."""
    prop = _input(modifier, identifier)
    if prop is not None:
        if not hasattr(prop, "attribute_name"):
            raise TypeError("Input %r does not support named attributes" % identifier)
        prop.attribute_name = attributeName
        prop.type = 'ATTRIBUTE'
    else:
        if identifier + "_use_attribute" not in modifier:
            raise TypeError("Input %r does not support named attributes" % identifier)
        modifier[identifier + "_attribute_name"] = attributeName
        modifier[identifier + "_use_attribute"] = True
    modifier.id_data.update_tag()


def getGnOutputAttribute(modifier, identifier):
    """Read the name used to store a field output on the generated geometry."""
    _socket(modifier, identifier, 'OUTPUT')
    if bpy.app.version >= (5, 2, 0):
        return getattr(modifier.properties.outputs, identifier).attribute_name
    return modifier[identifier + "_attribute_name"]


def setGnOutputAttribute(modifier, identifier, attributeName):
    """Set the output attribute name; an empty string disables storing it."""
    _socket(modifier, identifier, 'OUTPUT')
    if bpy.app.version >= (5, 2, 0):
        getattr(modifier.properties.outputs, identifier).attribute_name = attributeName
    else:
        key = identifier + "_attribute_name"
        if key not in modifier:
            raise TypeError("Output %r does not support storing an attribute" % identifier)
        modifier[key] = attributeName
    modifier.id_data.update_tag()
