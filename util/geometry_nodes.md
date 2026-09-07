# Geometry Nodes modifier compatibility

Use `util.geometry_nodes` for modifier input/output access. Blender 5.2 uses
RNA properties; earlier versions use ID properties. Only this module should
contain the version-specific access code. This follows the
[Blender 5.2 API change](https://developer.blender.org/docs/release_notes/5.2/python_api/#geometry-nodes).

```python
from ..util.geometry_nodes import (
    iterGnInputs, getGnInput, setGnInput,
    getGnInputAttribute, useAttributeForGnInput,
    getGnOutputAttribute, setGnOutputAttribute,
)

# Use the identifier from the node-tree interface, not a socket's position.
socket = next(s for s in iterGnInputs(modifier) if s.name == "Height")
setGnInput(modifier, socket.identifier, 12.0)
height = getGnInput(modifier, socket.identifier)

useAttributeForGnInput(modifier, socket.identifier, "building:height")
attribute_name = getGnInputAttribute(modifier, socket.identifier)
setGnInput(modifier, socket.identifier, 9.0)  # switches back to a literal value

setGnOutputAttribute(modifier, output_socket.identifier, "generated_height")
```

- Socket identifiers are stable across display-name changes. Resolve by name
  only when the asset contract guarantees a unique name; pass identifiers to
  the helpers. `iterGnInputs` skips panels and geometry sockets and supports
  both the pre-4.0 and current node-tree interfaces.
- Values may be numbers, booleans, strings, vectors, colors, or Blender data
  references supported by the socket. Blender performs type validation.
- `getGnInput` reads the stored literal value, not an evaluated field.
  Attribute bindings are read separately. `None` means no named-attribute
  binding; an empty string is an active binding with an empty name.
- Setting a value selects VALUE mode. Attribute binding selects ATTRIBUTE
  mode. Blender 5.2's separate LAYER-selection mode is not supported by the
  attribute helpers and is rejected explicitly when reading a binding.
- Setters affect the modifier instance and tag its owner for update. They do
  not change shared node-tree defaults. Evaluate the dependency graph when
  reading generated geometry after a change.
- Missing sockets raise an error rather than creating unused custom
  properties. Non-value inputs and unsupported attribute bindings also fail.
- `useAttributeForGnInput` remains importable from `util.blender` for existing
  callers. New code should import directly from `util.geometry_nodes`.

The same helpers apply to shared building node trees and trees generated for
individual buildings. They do not decide whether GN or Python volumes are used.

Run the compatibility checks with an installed Blender executable:

```text
blender --background --factory-startup --python-exit-code 1 --python tests/test_geometry_nodes.py
```

The test loads this helper directly, without enabling Blosm or opening user
files. It exercises real modifier values, attributes, socket validation, and
evaluated geometry.
