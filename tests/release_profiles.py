"""Release test selection and package contracts, independent of Blender."""
from pathlib import PurePosixPath
import tomllib
import zipfile

NEW = {
    'osm_progressive': 'check_osm_progressive.py',
    'osm': 'check_release_osm.py', 'gpx': 'check_release_gpx.py',
    'geometry_nodes': 'check_release_geometry_nodes.py', 'buildify': 'check_release_buildify.py',
    'terrain': 'check_release_terrain.py', 'overlay': 'check_release_overlay.py',
    'roofs': 'check_release_roofs.py', 'validation': 'check_release_validation.py',
    'export': 'check_release_export.py', 'baking': 'check_release_baking.py',
    'package': 'check_release_package.py', 'pml': 'check_release_pml.py',
    'base': 'check_release_base.py', 'pro': 'check_release_pro.py',
}
EXISTING = {
    'osm_progressive_ui': ('check_osm_progressive_ui.py', 'osm_ui_checks.json'),
    'direct': ('check_direct_gltf_import.py', 'correctness.json'),
    'operator': ('check_direct_gltf_operator.py', 'operator_check.json'),
    'progressive': ('check_progressive_import.py', 'checks.json'),
    'concurrent': ('check_concurrent_tiles.py', 'concurrent_checks.json'),
    'cesium_ion': ('check_cesium_ion.py', 'results.json'),
    'attribution': ('check_attribution.py', 'results.json'),
    'ui': ('check_progressive_ui.py', 'ui_checks.json'),
    'cesium_ion_ui': ('check_cesium_ion_ui.py', 'results.json'),
    'attribution_ui': ('check_attribution_ui.py', 'results.json'),
}
CORE = ['preflight', 'standalone', 'direct', 'operator', 'progressive', 'concurrent', 'osm_progressive',
        'cesium_ion', 'attribution',
        'osm', 'roofs', 'gpx', 'geometry_nodes', 'buildify', 'terrain', 'overlay',
        'validation', 'baking', 'export', 'pml']
PRO_ONLY = frozenset(('baking', 'export', 'pml'))
GUI_SUITES = ('ui', 'osm_progressive_ui', 'cesium_ion_ui', 'attribution_ui')
RELEASE_UI_SUITES = ('ui', 'cesium_ion_ui', 'attribution_ui')
SUITES = tuple(CORE + ['package', 'base', 'pro', *GUI_SUITES])

# Feature boundaries, not a second copy of the builder's exact file inventory.
# Shared properties/APE utilities remain valid in Base. Imagery and GN are common.
PRO_MODULES = ('action', 'building2', 'grammar', 'item', 'item_renderer', 'mpl',
               'pml', 'realistic', 'style', 'util.blender_extra', 'way')
PRO_SETUP = ('example_custom_textures', 'mpl_base', 'neoclassical', 'premium',
             'premium_default', 'realistic_dev', 'uv_only')
# Only operators registered by the current Pro build; legacy realistic tool
# classes still exist in source but are not exposed by either edition.
PRO_OPERATORS = ('create_materials', 'delete_materials', 'bake_to_single_material')


def select_suites(profile, requested=None):
    """Edition gates are mandatory even for a selected subset of feature tests."""
    if profile not in ('general', 'base', 'pro'):
        raise ValueError(f'Unknown profile: {profile}')
    wanted = list(dict.fromkeys(CORE if requested is None else requested))
    unknown = set(wanted) - set(SUITES)
    if unknown:
        raise ValueError(f'Unknown suites: {sorted(unknown)}')
    for dedicated in ('base', 'pro'):
        if dedicated in wanted and profile != dedicated:
            raise ValueError(f'{dedicated} suite requires --profile {dedicated}')
    excluded = []
    if profile == 'base':
        if requested is not None and PRO_ONLY.intersection(wanted):
            raise ValueError(f'Pro-only suites requested for Base: {sorted(PRO_ONLY.intersection(wanted))}')
        excluded = [dict(suite=name, status='not_applicable', reason='Pro-only feature')
                    for name in wanted if name in PRO_ONLY]
        wanted = [name for name in wanted if name not in PRO_ONLY]
    if profile != 'general':
        wanted = ['package', profile] + [name for name in wanted if name not in ('package', profile)]
    return wanted, excluded


def inspect_candidate(path):
    """Validate a root-layout extension ZIP before Blender or extraction sees it."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for info in archive.infolist():
            # ZipInfo normalizes Windows separators when reading the central
            # directory. Check the original spelling before that normalization.
            name = info.orig_filename
            parts = PurePosixPath(name)
            if (not name or parts.is_absolute() or '..' in parts.parts or '\\' in name
                    or ':' in name or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe ZIP entry: {name}')
        if not {'blender_manifest.toml', '__init__.py'} <= set(names):
            raise ValueError('Build profiles require an extension ZIP with its manifest at the root')
        manifest = tomllib.loads(archive.read('blender_manifest.toml').decode('utf-8'))
        if manifest.get('id') != 'blosm' or manifest.get('type') != 'add-on':
            raise ValueError('Expected a Blosm add-on extension')
        return manifest, names
