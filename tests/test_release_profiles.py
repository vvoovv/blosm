"""Release selection must not omit build gates or load checkout runtime code."""
from io import BytesIO
import unittest
import zipfile

from release_profiles import CORE, GUI_SUITES, RELEASE_UI_SUITES, PRO_ONLY, inspect_candidate, select_suites
from run_candidate_suite import SourceIsolation, SOURCE, TESTS


class ReleaseProfileChecks(unittest.TestCase):
    def test_base_runs_common_features_and_explicitly_excludes_pro(self):
        selected, excluded = select_suites('base')
        self.assertEqual(selected[:2], ['package', 'base'])
        self.assertEqual(set(selected[2:]), set(CORE) - PRO_ONLY)
        self.assertTrue({'osm', 'gpx', 'buildify', 'overlay', 'concurrent',
                         'cesium_ion', 'attribution', 'tile_transforms'} <= set(selected))
        self.assertEqual({row['suite'] for row in excluded}, PRO_ONLY)
        self.assertTrue(all(row['status'] == 'not_applicable' for row in excluded))

    def test_pro_reuses_the_entire_general_catalog(self):
        general, _ = select_suites('general')
        selected, excluded = select_suites('pro')
        self.assertEqual(selected, ['package', 'pro', *general])
        self.assertEqual(excluded, [])

    def test_new_ui_checks_run_as_gui_suites_with_edition_gates(self):
        self.assertTrue({'cesium_ion_ui', 'attribution_ui'} <= set(RELEASE_UI_SUITES))
        self.assertTrue(set(RELEASE_UI_SUITES) <= set(GUI_SUITES))
        for profile in ('base', 'pro'):
            selected, excluded = select_suites(profile, RELEASE_UI_SUITES)
            self.assertEqual(selected, ['package', profile, *RELEASE_UI_SUITES])
            self.assertEqual(excluded, [])

    def test_selected_features_cannot_bypass_edition_gates(self):
        for profile in ('base', 'pro'):
            selected, _ = select_suites(profile, ['osm', 'osm', profile, 'package'])
            self.assertEqual(selected, ['package', profile, 'osm'])
        for profile, requested in (('general', ['base']), ('base', ['pro']),
                                   ('pro', ['base']), ('base', ['baking']),
                                   ('base', ['export']), ('base', ['pml'])):
            with self.subTest(profile=profile, requested=requested), self.assertRaises(ValueError):
                select_suites(profile, requested)

    def test_checkout_imports_are_rejected_but_test_helpers_are_allowed(self):
        finder = SourceIsolation()
        # These calls inspect module locations without executing addon code.
        with self.assertRaisesRegex(ImportError, 'checkout module'):
            finder.find_spec(SOURCE.name, [str(SOURCE.parent)])
        with self.assertRaisesRegex(ImportError, 'checkout module'):
            finder.find_spec('downloads', [str(SOURCE)])
        self.assertIsNone(finder.find_spec('release_profiles', [str(TESTS)]))

    def test_candidate_layout_and_unsafe_paths_fail_before_installation(self):
        def archive(*extra):
            stream = BytesIO()
            with zipfile.ZipFile(stream, 'w') as output:
                output.writestr('blender_manifest.toml', 'id = "blosm"\ntype = "add-on"\n')
                output.writestr('__init__.py', '')
                for path in extra:
                    entry = zipfile.ZipInfo(path)
                    # ZipInfo normalizes backslashes on Windows; simulate the
                    # raw entry a ZIP produced on another platform can contain.
                    entry.filename = path
                    output.writestr(entry, '')
            stream.seek(0)
            return stream
        manifest, _ = inspect_candidate(archive())
        self.assertEqual(manifest['id'], 'blosm')
        for path in ('../escape.py', '/escape.py', 'folder\\escape.py', 'file:stream'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Unsafe ZIP'):
                inspect_candidate(archive(path))
        legacy = BytesIO()
        with zipfile.ZipFile(legacy, 'w') as output:
            output.writestr('blosm/__init__.py', '')
        legacy.seek(0)
        with self.assertRaisesRegex(ValueError, 'manifest at the root'):
            inspect_candidate(legacy)


if __name__ == '__main__':
    unittest.main()
