"""Explicit release inventory. Run with ordinary Python 3.11+, not inside Blender."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from release_support import REPO, DEFAULT_ARTIFACTS, catalog, osm_path, real_fixture
from release_process import owned_process

from release_profiles import NEW, EXISTING, CORE, SUITES, GUI_SUITES, select_suites, inspect_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--profile-root', type=Path,
                        help='Isolated Blender profile root owned by this chat/run')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--profile', choices=('general', 'base', 'pro'), default='general')
    parser.add_argument('--suite', action='append', choices=SUITES)
    parser.add_argument('--artifacts', type=Path,
                        default=DEFAULT_ARTIFACTS / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--osm-root', type=Path, default=REPO.parent / 'osm_extracts')
    parser.add_argument('--fixture-root', type=Path, default=DEFAULT_ARTIFACTS / 'fixtures')
    parser.add_argument('--assets', type=Path, help='Pro asset directory (required for export suite)')
    parser.add_argument('--candidate-zip', type=Path, help='Candidate extension ZIP (required for Base/Pro profiles)')
    parser.add_argument('--helsinki', type=Path, help='Optional downloaded real tile fixtures')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800, help='Seconds per suite')
    args = parser.parse_args()
    # CLI paths are relative to the caller. Resolve them before changing the
    # subprocess working directory to an isolated candidate-test directory.
    for name in ('blender', 'python', 'artifacts', 'osm_root', 'fixture_root', 'profile_root',
                 'assets', 'candidate_zip', 'helsinki'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    try:
        selected, excluded = select_suites(args.profile, args.suite)
        if args.profile != 'general':
            if not args.candidate_zip or not args.candidate_zip.is_file():
                parser.error('--profile base/pro requires --candidate-zip pointing to a file')
            inspect_candidate(args.candidate_zip)
        elif args.candidate_zip or 'package' in selected:
            parser.error('Use --profile base or pro to test a candidate ZIP')
    except ValueError as error:
        parser.error(str(error))
    # Never overwrite an earlier result with stale JSON or a partial run.
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if any(args.artifacts.iterdir()):
        parser.error('Choose an empty --artifacts directory; previous output must not be reused')
    if not args.blender.is_file():
        parser.error(f'Blender executable missing: {args.blender}')
    report = dict(started=datetime.now(timezone.utc).isoformat(), source=str(REPO),
                  blender=str(args.blender), python=str(args.python), profile=args.profile,
                  selection='full_headless' if args.suite is None else 'selected',
                  excluded_suites=excluded, suites=[])
    if args.candidate_zip:
        original = args.candidate_zip
        args.candidate_zip = args.artifacts / 'candidate.zip'
        shutil.copy2(original, args.candidate_zip)
        report['candidate'] = dict(sha256=hashlib.sha256(args.candidate_zip.read_bytes()).hexdigest(),
                                   snapshot='candidate.zip')
    for key, command in (('commit', ['git', 'rev-parse', 'HEAD']),
                         ('working_tree', ['git', 'status', '--short'])):
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        report[key] = result.stdout.strip() if result.returncode == 0 else 'unavailable'
    report['test_source_sha256'] = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((REPO / 'tests').rglob('*')) if p.suffix in ('.py', '.toml')}
    source_hash = hashlib.sha256()
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=REPO, capture_output=True, check=True)
    for relative in sorted(tracked.stdout.decode('utf-8').split('\0')):
        path = REPO / relative
        if relative and path.is_file():
            source_hash.update(relative.encode('utf-8') + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    report['tracked_source_sha256'] = source_hash.hexdigest()
    version_env = os.environ.copy()
    version_profile = (args.profile_root or args.artifacts/'profiles')/'version'
    for kind in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'EXTENSIONS', 'DATAFILES'):
        location = version_profile/kind.lower()
        location.mkdir(parents=True, exist_ok=True)
        version_env['BLENDER_USER_'+kind] = str(location)
    report['blender_version'] = subprocess.check_output([str(args.blender), '--factory-startup',
        '--disable-autoexec', '--version'], env=version_env, text=True).splitlines()[0]
    gate_failed = False
    for name in selected:
        started = time.monotonic()
        row = dict(suite=name, status='failed')
        folder = args.artifacts / name
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if gate_failed:
                row['status'] = 'blocked'
                raise RuntimeError('Required package/edition gate failed')
            if name == 'preflight':
                for entry in catalog(args.extended):
                    osm_path(args, entry)
                for kind in ('gpx', 'gn2d'):
                    real_fixture(args, kind)
                row['status'] = 'passed'
            else:
                extra = []
                if name in NEW:
                    extra += ['--osm-root', str(args.osm_root), '--fixture-root', str(args.fixture_root)]
                    if args.assets:
                        extra += ['--assets', str(args.assets)]
                    if args.candidate_zip:
                        extra += ['--candidate-zip', str(args.candidate_zip)]
                    if args.extended:
                        extra.append('--extended')
                if args.helsinki and name in ('direct', 'progressive', 'concurrent', 'tile_transforms'):
                    if len(list(args.helsinki.glob('*.b3dm'))) < 3:
                        raise FileNotFoundError('Helsinki fixture must contain at least three B3DM files')
                    extra += ['--helsinki', str(args.helsinki)]
                if name == 'standalone' and args.profile == 'general':
                    command = [str(args.python), '-B', '-m', 'unittest', 'discover',
                               '-s', str(REPO / 'tests'), '-p', 'test_*.py', '-v']
                else:
                    script = 'standalone' if name == 'standalone' else NEW.get(name) or EXISTING[name][0]
                    if name == 'osm_progressive_ui' and args.assets:
                        extra += ['--assets', str(args.assets)]
                    if name == 'standalone':
                        command = [str(args.python), '-B']
                    else:
                        command = [str(args.blender)] + ([] if name in GUI_SUITES else ['--background'])
                        if name in ('concurrent', 'ui', 'overlay', 'cesium_ion_ui'):
                            # Blender's extension policy governs localhost and
                            # the mocked imagery download path as well.
                            command.append('--online-mode')
                        command += ['--factory-startup', '--disable-autoexec', '--python-exit-code', '1', '--python']
                    if args.profile == 'general':
                        command += [str(REPO / 'tests' / script), '--', '--artifacts', str(folder), *extra]
                    else:
                        command += [str(REPO / 'tests/run_candidate_suite.py')]
                        if name != 'standalone':
                            command.append('--')
                        command += ['--candidate-zip', str(args.candidate_zip), '--profile', args.profile,
                                    '--suite', script, '--artifacts', str(folder), '--', *extra]
                env = os.environ.copy()
                for variable in ('BLOSM_TEST_ROOT', 'BLOSM_TEST_PACKAGE', 'BLOSM_TEST_PROFILE'):
                    env.pop(variable, None)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                profile = (args.profile_root / name) if args.profile_root else (folder / 'profile')
                profile.mkdir(parents=True, exist_ok=True)
                env['BLENDER_USER_RESOURCES'] = str(profile)
                temporary = folder / 'temporary'
                temporary.mkdir(parents=True, exist_ok=True)
                for variable in ('TMP', 'TEMP', 'TMPDIR'):
                    env[variable] = str(temporary)
                for kind in ('CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS'):
                    location = profile / kind.lower()
                    location.mkdir(parents=True, exist_ok=True)
                    env['BLENDER_USER_' + kind] = str(location)
                row['command'] = command
                print(f'RUN {name}', flush=True)
                startup = None
                if os.name == 'nt':
                    startup = subprocess.STARTUPINFO()
                    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup.wShowWindow = 0
                with (folder / 'console.log').open('w', encoding='utf-8') as log:
                    with owned_process(command, cwd=folder if args.profile != 'general' else REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, startupinfo=startup) as process:
                        try:
                            row['exit_code'] = process.wait(timeout=args.timeout)
                        except subprocess.TimeoutExpired:
                            raise TimeoutError(f'{name} exceeded {args.timeout} seconds') from None
                if row['exit_code'] != 0:
                    raise RuntimeError(f'Exit {row["exit_code"]}; see {folder / "console.log"}')
                if name in NEW or name in EXISTING:
                    result_path = folder / ('results.json' if name in NEW else EXISTING[name][1])
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                    if result.get('status') != 'passed':
                        raise AssertionError(f'Nonpassing status in {result_path}')
                row['status'] = 'passed'
        except Exception as error:
            row['error'] = str(error)
            if args.profile != 'general' and name in ('package', args.profile):
                gate_failed = True
        row['seconds'] = round(time.monotonic() - started, 3)
        report['suites'].append(row)
        report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['suites']) else 'failed'
        (args.artifacts / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'{row["status"].upper()} {name} ({row["seconds"]}s)', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
