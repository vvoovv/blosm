# OpenStreetMap Importer for Blender

[![](https://raw.githubusercontent.com/wiki/vvoovv/blender-osm/images/import_osm.png)](https://gumroad.com/l/blender-osm)

## Introduction

The addon imports a file in the OpenStreetMap format (.osm).

There are free and [paid](https://gumroad.com/l/blender-osm) (2.85$) versions of the addon.

[![](https://raw.githubusercontent.com/wiki/vvoovv/blender-osm/images/paid_vs_free.png)](https://gumroad.com/l/blender-osm)

The free version is 2D only. Also a popup window is shown after the import finished. A mouse click closes the popup window and brings you to the [webpage](https://gumroad.com/l/blender-osm) to buy the paid version for just 2.85$.

The [paid](https://gumroad.com/l/blender-osm) version supports 3D buildings and doesn't show you an annoying popup window.

By [buying](https://gumroad.com/l/blender-osm) the paid version you support the addon development.

The following items can be imported by the addon:
* Buildings. Building height, number of floors are used to create the final scene. Composition into 3D parts for a building with the complex structure is also processed.
* Water objects (rivers and lakes). Imported as polygons. Coastlines for seas and oceans are importes as edges.
* Highways, paths and railways. Imported as edges.
* Vegetation (forests, grass, scrubs). Imported as polygons.

Polygons with holes are supported!

## Installation
The addon requires at least Blender 2.76, but it makes sense to use the latest version of Blender.

[Buy](https://gumroad.com/l/blender-osm) the addon for just 2.85$ or download a free version from [here](https://github.com/vvoovv/blender-osm/archive/master.zip). In either case you will get a zip archive. Do not unpack it! Install it via the usual Blender way:
* Delete the previous version of the addon if you have one:
    * _File → User Preferences... → Addons_
    * Type _osm_ in the search box in the top left corner of the _Addons_ tab to find the addon
    * Press _Remove_ button in the GUI box with the header _Import-Export: Import OpenStreetMap (.osm)_
* _File → User Preferences... → Addons → Install from File..._
* Find the zip archive in your file system and press _Install from File..._ button
* Enable the addon by checking the _Enable an addon_
* Press _Save User Settings_ in the _Blender User Preferences_ window

## Usage
* _File → Import → OpenStreetMap (.osm)_
* Find an OpenStreetMap file in your file system
* Press _Import OpenStreetMap_ button
* If you have the free version, a popup window shows up after the import finished. A mouse click closes the popup window and brings you to the [webpage](https://gumroad.com/l/blender-osm) to buy the paid version for just 2.85$.
* The [paid](https://gumroad.com/l/blender-osm) version doesn't show you the popup and brings you directly to the imported scene.

For detailed instructions, limitations, tips and tricks see the [Documentation](https://github.com/vvoovv/blender-osm/wiki/Documentation).


## Links
* [Paid version](https://gumroad.com/l/blender-osm)
* [Documentation](https://github.com/vvoovv/blender-osm/wiki/Documentation)
* twitter: [@prokitektura](https://twitter.com/prokitektura)
* blenderartists.org: [thread](http://blenderartists.org/forum/showthread.php?334508-Addon-Import-OpenStreetMap-(-osm))
<br>post there questions, nice renderings created with the addon and your experience with the addon
* bugs and feature requests: [issues](https://github.com/vvoovv/blender-osm/issues)
