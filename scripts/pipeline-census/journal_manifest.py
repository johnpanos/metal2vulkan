#!/usr/bin/env python3
"""Turn a NativeMetal frontend pipeline journal (pipelines.log: its R lines, one per render pipeline
creation, and S lines, one per function specialization) into a command-replay manifest (version 1)
with the tables `pipeline-census` reads: `libraries`, `functions` and `renderPipelines`.

    journal_manifest.py JOURNAL --library ID=FILE [--library ID=FILE ...] --out MANIFEST.json

FILE paths are written as given, relative to the manifest's directory when they are relative.
The journal names no library, so the functions carry none and `pipeline-census` looks each up by
name in the libraries in the order given. Identical pipelines (same functions, constants and
descriptor) are kept once; `count` says how many creations the journal had of each, `label` is
the caller's label of the first.

Journal fields (NativeBridgeMetalFrontend.mm): `samples=`, `topo=`, `ds=<depth>/<stencil>`,
`label=`, `c<i>=format:mask:blending:srcRGB:dstRGB:rgbOp:srcA:dstA:alphaOp`,
`a<i>=format:offset:buffer:stride:step:rate`, then `| v NAME CONSTANTS | f NAME CONSTANTS`, each
constant `name:type:hex` or `#index:type:hex`.

GUEST DATA: the journal and the manifest describe the consumer's shaders; keep them under a
gitignored directory (.cache/), never committed.
"""
import argparse
import json
import os
import re
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('journal')
ap.add_argument('--library', action='append', default=[], help='ID=FILE, in lookup order')
ap.add_argument('--out', required=True)
ap.add_argument('--include-refused', action='store_true', help='keep pipelines the device refused (ok=0)')
args = ap.parse_args()


def constants(text):
    out = []
    for item in text.split():
        key, ty, hexbytes = item.rsplit(':', 2)
        c = {'type': int(ty), 'hex': hexbytes}
        if key.startswith('#'):
            c['index'] = int(key[1:])
        else:
            c['name'] = key
        out.append(c)
    return out


functions, function_ids, pipelines, keys = {}, {}, {}, {}
lines = skipped = 0


def function(name, consts):
    key = (name, consts)
    if key not in function_ids:
        fid = 'f%d' % len(function_ids)
        function_ids[key] = fid
        functions[fid] = {'name': name, 'constants': constants(consts)}
    return function_ids[key]


for line in open(args.journal, errors='replace'):
    line = line.rstrip('\n')
    if line.startswith('S ') and ' ok=1 ' in line:
        m = re.search(r' name=(\S+) k=(.*?) \| ', line + ' | ')
        if m:
            function(m.group(1), m.group(2).strip())
        continue
    if not line.startswith('R '):
        continue
    lines += 1
    parts = line.split(' | ')
    head = parts[0]
    if ' ok=1 ' not in head + ' ' and not args.include_refused:
        skipped += 1
        continue
    stages = {}
    for p in parts[1:]:
        m = re.match(r'([vf]) (\S+) ?(.*)$', p)
        if m:
            stages[m.group(1)] = (m.group(2), m.group(3).strip())
    if 'v' not in stages or stages['v'][0] == '-':
        skipped += 1
        continue
    fields = dict(re.findall(r'(\S+?)=(\S+)', head))
    p = {'vertex': function(*stages['v'])}
    if 'f' in stages and stages['f'][0] != '-':
        p['fragment'] = function(*stages['f'])
    else:
        p['fragment'] = None
    depth, stencil = (fields.get('ds', '0/0').split('/') + ['0'])[:2]
    p.update(samples=int(fields.get('samples', 1)), topology=int(fields.get('topo', 0)),
             depth=int(depth), stencil=int(stencil))
    colors = []
    for i in range(8):
        c = fields.get('c%d' % i)
        if c is None:
            colors.append(None)
            continue
        v = [int(x) for x in c.split(':')]
        colors.append({'format': v[0], 'mask': v[1], 'blending': bool(v[2]), 'srcRGB': v[3], 'dstRGB': v[4],
                       'rgbOp': v[5], 'srcA': v[6], 'dstA': v[7], 'alphaOp': v[8]})
    while colors and colors[-1] is None:
        colors.pop()
    p['colors'] = colors
    attributes, layouts = [], {}
    for i in range(31):
        a = fields.get('a%d' % i)
        if a is None:
            continue
        fmt, offset, buf, stride, step, rate = (int(x) for x in a.split(':'))
        attributes.append({'index': i, 'format': fmt, 'offset': offset, 'buffer': buf})
        layouts[buf] = {'index': buf, 'stride': stride, 'step': step, 'rate': rate}
    if attributes:
        p['vertexDescriptor'] = {'attributes': attributes, 'layouts': [layouts[k] for k in sorted(layouts)]}
    key = json.dumps(p, sort_keys=True)
    if key in keys:
        pipelines[keys[key]]['count'] += 1
        continue
    pid = 'r%d' % len(pipelines)
    keys[key] = pid
    p['label'] = fields.get('label', '-')
    p['count'] = 1
    pipelines[pid] = p

libraries = {}
for spec in args.library:
    lid, _, path = spec.partition('=')
    if not path:
        sys.exit('--library takes ID=FILE')
    libraries[lid] = path
manifest = {'version': 1, 'libraries': libraries, 'functions': functions, 'renderPipelines': pipelines}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(manifest, f, indent=1)
print('%s: %d render pipeline lines, %d kept as %d distinct pipelines over %d functions (%d skipped)'
      % (args.out, lines, lines - skipped, len(pipelines), len(functions), skipped))
