#!/usr/bin/env python3
"""Turn a NativeMetal pipeline capture (NGPU_CAPTURE: one directory per process, capture.jsonl and
lib-<sha>.metallib; PipelineCapture.h in NativeMetal) into a command-replay manifest (version 1)
with the tables `pipeline-census` reads: `libraries`, `functions`, `renderPipelines`,
`computePipelines`, and `draws` (each distinct draw state: its pipeline, pass, bindings by slot,
depth/stencil and raster state).

    capture_manifest.py CAPTURE_DIR [CAPTURE_DIR ...] --out MANIFEST.json

The manifest names the libraries by their paths relative to its own directory. Identical
pipelines (same functions and descriptor) are kept once, across processes; `count` says how many
creations the captures had of each. Pipelines the device refused are left out.

GUEST DATA: the libraries are Apple's (the render server's own shaders); keep the capture and the
manifest under a gitignored directory (.cache/), never committed.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('captures', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--xr-as-f16', action='store_true',
                help="as NativeMetal 4101648's frontend does: the extended-range 10-bit formats (550-555) of pipelines, pass attachments "
                     "and textures not over a buffer become RGBA16Float (115), for a capture made before it")
args = ap.parse_args()
out_dir = os.path.dirname(os.path.abspath(args.out))
libraries, functions, render, compute, draws = {}, {}, {}, {}, {}
keys = {}
lines = 0
for cap in args.captures:
    numbers = {}   # this process's render numbers -> manifest ids
    for line in open(os.path.join(cap, 'capture.jsonl'), errors='replace'):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue   # a line cut short by the process's end
        lines += 1
        if 'library' in r and 'file' in r:
            if not os.path.exists(os.path.join(cap, r['file'])):
                print('%s: library %s (%d bytes) not pulled; left out' % (cap, r['library'], r.get('bytes', 0)))
                continue
            libraries[r['library']] = os.path.relpath(os.path.join(cap, r['file']), out_dir)
        elif 'function' in r and 'name' in r:
            f = {'name': r['name'], 'constants': r.get('constants', [])}
            if r.get('library'):
                f['library'] = r['library']
            functions[r['function']] = f
        elif 'render' in r and 'draw' not in r:
            if not r.get('ok'):
                continue
            p = {k: v for k, v in r.items() if k not in ('render', 'ok', 'label')}
            key = 'render ' + json.dumps(p, sort_keys=True)
            if key not in keys:
                keys[key] = 'r%d' % len(render)
                p['label'] = r.get('label')
                p['count'] = 0
                render[keys[key]] = p
            render[keys[key]]['count'] += 1
            numbers[r['render']] = keys[key]
        elif 'compute' in r:
            if not r.get('ok'):
                continue
            key = 'compute ' + r['function']
            if key not in keys:
                keys[key] = 'c%d' % len(compute)
                compute[keys[key]] = {'function': r['function'], 'label': r.get('label'), 'count': 0}
            compute[keys[key]]['count'] += 1
        elif 'draw' in r:
            pid = numbers.get(r['render'])
            if pid is None:
                continue
            d = {k: v for k, v in r.items() if k != 'draw'}
            d['render'] = pid
            key = 'draw ' + json.dumps(d, sort_keys=True)
            if key not in keys:
                keys[key] = 'd%d' % len(draws)
                draws[keys[key]] = d
for f in functions.values():
    if f.get('library') not in libraries:
        f.pop('library', None)
def xr(f):
    return 115 if args.xr_as_f16 and 550 <= f <= 555 else f


if args.xr_as_f16:
    for p in render.values():
        for c in p.get('colors', []):
            if c:
                c['format'] = xr(c['format'])
    for d in draws.values():
        for c in d.get('pass', {}).get('colors', []):
            if c:
                c['format'] = xr(c['format'])
        for slot, b in d.get('slots', {}).items():
            if slot[1:2] == 't' and not b.get('over'):
                b['format'] = xr(b['format'])
manifest = {'version': 1, 'libraries': libraries, 'functions': functions, 'renderPipelines': render,
            'computePipelines': compute, 'draws': draws}
with open(args.out, 'w') as f:
    json.dump(manifest, f, indent=1)
print('%s: %d records -> %d libraries, %d functions, %d render and %d compute pipelines, %d draw states'
      % (args.out, lines, len(libraries), len(functions), len(render), len(compute), len(draws)))
