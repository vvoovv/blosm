# Blender extension release

The release branch uses Blender's extension format and requires Blender 4.5 or
newer. `blender_manifest.toml` is the source of version and package metadata.
Install the release ZIP through **Preferences > Get Extensions > Install from
Disk**, then open the Blosm tab in the 3D View sidebar.

## Build and validate

Release tooling and Base/Pro inventories live in the sibling
[`blosm_build` repository](../../blosm_build/README.md). This repository's
`blender_manifest.toml` contains runtime metadata only; do not add `[build]` here.
Direct builds with Blender's default exclusions are not release packages.

Run these commands from the Blosm repository root with Python 3.11+.
Filesystem paths below are relative to that root; adjust the example Blender
executable location to your setup.

```powershell
$blender = '../tools/blender/blender.exe'

# Official build from committed source, with tests before publication:
python ../blosm_build/build_extension.py --source . --ref release --edition both --test --blender $blender

# Preview uncommitted changes:
python ../blosm_build/build_extension.py --source . --working-tree --edition both --test --blender $blender
```

Output defaults to `../artifacts/blosm-releases`. Each run has its
own logs, input hashes, provenance report and `packages/` directory. With
`--test`, standalone Blosm tests run before candidate packaging and the actual
Base and Pro ZIPs are installed and exercised in isolated Blender profiles.
No release ZIPs are published to `packages/` if any required check fails.
Without `--test`, manifest, ZIP and edition-inventory validation still run.

The install checker accepts `--edition base` or `--edition pro` (default: pro).
It checks preferences, namespaces, offline behavior, local OSM, GLB/B3DM, and
Pro PML when applicable. The broader release tests under `tests/` can additionally
exercise real datasets, UI and external Pro asset packs; see the
[General, Base and Pro test profiles](../tests/README.md).

The 2.8.1 release notes are in [docs/releases/2.8.1.md](releases/2.8.1.md).
Both build profiles include the `cesium_ion` and `attribution` headless suites.
The release qualification entry point also selects `cesium_ion_ui` and
`attribution_ui` alongside the existing 3D Tiles UI suite. These scripts use the
installed candidate package under Base/Pro profiles, never checkout runtime code.

## Runtime conventions

- Internal and vendored imports remain relative to the installed package.
  Blender's `sys.path` and unrelated global modules are not modified. The
  standalone download subprocess has its own isolated Python environment.
- Preferences use the root `__package__`, including the extension repository.
- Downloads check `bpy.app.online_access`. Revoking access closes an active
  download transport when it is next polled or submitted to. Local input and
  cached imagery remain usable offline.
- Every OSM import, including local files and every rendering mode, requires an
  existing data directory in the add-on preferences. Validation happens before
  asset checks, directory creation or downloading. OSM downloads go into its
  `osm` subdirectory; an empty setting never falls back to the addon directory.
- For other downloads with no custom data directory, downloads go to
  `bpy.utils.extension_path_user`, outside the installed extension. User-selected
  asset directories remain external to the extension.
- 3D Tiles rebase private glTF node data before single-precision conversion;
  they do not patch Blender's shared glTF importer classes.
- Custom setup scripts load as `<extension>.setup._user_script`. Use relative
  imports such as `from ..app.blender import app` or
  `from ..manager import BaseManager`; old top-level imports such as
  `from app.blender import app` must be updated.
- Regenerated ANTLR parser files must retain relative imports of the bundled
  `pml.antlr4` runtime. No dependency installation is performed at runtime.

This is technical extension compatibility. Submission to extensions.blender.org
is a separate publishing process with its own content and review requirements.

References: [extension manifest and packaging](https://docs.blender.org/manual/en/latest/advanced/extensions/getting_started.html),
[add-on migration](https://docs.blender.org/manual/en/latest/advanced/extensions/addons.html),
[Blender add-on guidelines](https://developer.blender.org/docs/handbook/addons/guidelines/).
