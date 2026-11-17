"""Launch the GUI benchmark without touching the normal Blender user profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ctypes
import hashlib
from ctypes import wintypes

from release_process import owned_process

parser = argparse.ArgumentParser()
parser.add_argument('--blender', type=Path, default=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'))
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--profile', type=Path, required=True)
parser.add_argument('--debug-graph', action='store_true')
parser.add_argument('--prototype-check', action='store_true')
args, extra = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
assert '--' not in extra
args.artifacts.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
for kind in ('RESOURCES', 'CONFIG', 'SCRIPTS', 'EXTENSIONS', 'DATAFILES'):
    directory = args.profile/kind.lower()
    directory.mkdir(parents=True, exist_ok=True)
    environment['BLENDER_USER_'+kind] = str(directory)
temporary = args.artifacts/'temp'
temporary.mkdir(exist_ok=True)
environment['TEMP'] = environment['TMP'] = str(temporary)
seed = [extra[extra.index('--seed-scene')+1]] if '--seed-scene' in extra else []
debug = ['--debug-depsgraph-time'] if args.debug_graph else []
script = 'check_tiles_profile_prototype.py' if args.prototype_check else 'benchmark_tiles_progressive.py'
background = ['--background'] if args.prototype_check else []
command = [str(args.blender), *background, '--factory-startup', '--disable-autoexec', *debug, *seed, '--python-exit-code', '1',
           '--python', str(root/'tests'/script), '--',
           '--artifacts', str(args.artifacts), *extra]
provenance = dict(command=command, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  source=str(root), profile=str(args.profile),
                  benchmark_sha256=hashlib.sha256((root/'tests/benchmark_tiles_progressive.py').read_bytes()).hexdigest())
(args.artifacts/'launch.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
start=time.perf_counter()
memory=[]


def memory_usage(process):
    class Counters(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('faults',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('peak_rss','rss','peak_pool','pool',
            'peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    counters=Counters()
    counters.size=ctypes.sizeof(counters)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    if not api.GetProcessMemoryInfo(int(process._handle),ctypes.byref(counters),counters.size):
        return None
    return dict(seconds=time.perf_counter()-start,rss=counters.rss,private=counters.private)
with (args.artifacts/'console.log').open('w',encoding='utf-8') as log:
    with owned_process(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT) as process:
        try:
            while process.poll() is None:
                sample=memory_usage(process)
                if sample:
                    memory.append(sample)
                (args.artifacts/'memory.json').write_text(json.dumps(memory),encoding='utf-8')
                if time.perf_counter()-start>1600:
                    raise subprocess.TimeoutExpired(command,1600)
                time.sleep(1)
            code=process.returncode
        except subprocess.TimeoutExpired:
            raise RuntimeError('Benchmark process exceeded its deadline')
report_path=args.artifacts/('prototype_parity.json' if args.prototype_check else 'profile.json')
report=json.loads(report_path.read_text()) if report_path.is_file() else {}
print(args.artifacts.name,code,report.get('status','no report'),round(time.perf_counter()-start,3),flush=True)
if code or report.get('status')!='passed':
    (args.artifacts/'failure.json').write_text(json.dumps(dict(exit_code=code,
       elapsed=time.perf_counter()-start,memory=memory[-1:])),encoding='utf-8')
    raise SystemExit(1)
