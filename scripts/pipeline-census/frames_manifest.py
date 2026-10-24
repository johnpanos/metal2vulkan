#!/usr/bin/env python3
"""Turn one process's NativeMetal frame capture (NGPU_CAPTURE with NGPU_CAPTURE_FRAMES: capture.jsonl,
frames.jsonl and their blob and library files) into a command-replay manifest (version 1) that
replays the captured command buffers in commit order.

    frames_manifest.py CAPTURE_DIR --out MANIFEST.json [--readback last|all|none]

    command-replay run --manifest MANIFEST.json --out DIR --device-type cpu|gpu --shader-cache DIR \\
        --metal-level newest --unorm8-filter device

Pipelines keep the capture's numbers (r<N>, c<N>); resources keep their ids (b, t, s, d). Each
command buffer starts with the uploads of every shared buffer whose bytes changed since it was
last recorded. `--readback last` (the default) reads back the color attachments of the last
render pass, the frame the window ended on; `all` every texture a pass wrote.

GUEST DATA: the capture is the render server's (Apple's shaders, the screen's contents): keep it
and the manifest under a gitignored directory (.cache/), never committed.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('capture')
ap.add_argument('--out', required=True)
ap.add_argument('--readback', choices=('last', 'all', 'none'), default='last')
args = ap.parse_args()
out_dir = os.path.dirname(os.path.abspath(args.out))
rel = lambda name: os.path.relpath(os.path.join(args.capture, name), out_dir)


def lines(name):
    path = os.path.join(args.capture, name)
    if not os.path.exists(path):
        return
    for line in open(path, errors='replace'):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass   # a line cut short by the process's end


m = {'version': 1, 'libraries': {}, 'functions': {}, 'renderPipelines': {}, 'computePipelines': {},
     'buffers': {}, 'textures': {}, 'views': {}, 'samplers': {}, 'depthStencilStates': {}, 'commandBuffers': []}
for r in lines('capture.jsonl'):
    if 'library' in r and 'file' in r:
        m['libraries'][r['library']] = rel(r['file'])
    elif 'function' in r:
        f = {'name': r['name'], 'constants': r.get('constants', [])}
        if r.get('library'):
            f['library'] = r['library']
        m['functions'][r['function']] = f
    elif 'render' in r and 'draw' not in r and r.get('ok'):
        m['renderPipelines']['r%d' % r['render']] = {k: v for k, v in r.items() if k not in ('render', 'ok')}
    elif 'compute' in r and r.get('ok'):
        m['computePipelines']['c%d' % r['compute']] = {'function': r['function'], 'label': r.get('label')}
for f in m['functions'].values():
    if f.get('library') not in m['libraries']:
        f.pop('library', None)
written = []
for r in lines('frames.jsonl'):
    if 'commandBuffer' in r:
        cb = {'uploads': [{'buffer': u['buffer'], 'file': rel(u['file'])} for u in r.get('uploads', []) if u.get('file')],
              'encoders': r.get('encoders', [])}
        m['commandBuffers'].append(cb)
        for e in cb['encoders']:
            if 'render' in e:
                written.append([c['texture'] for c in e['render'].get('colors', []) if c])
    elif 'buffer' in r and 'length' in r:
        m['buffers'][r['buffer']] = {'length': r['length'], 'options': r.get('options', 0)}
    elif 'texture' in r and 'format' in r:
        t = {k: v for k, v in r.items() if k not in ('texture', 'surface', 'contents')}
        if 'file' in t:
            t['file'] = rel(t['file'])
        m['textures'][r['texture']] = t
    elif 'view' in r:
        m['views'][r['view']] = {k: v for k, v in r.items() if k != 'view'}
    elif 'sampler' in r:
        m['samplers'][r['sampler']] = r.get('descriptor', {})
    elif 'depthStencil' in r:
        m['depthStencilStates'][r['depthStencil']] = r.get('descriptor', {})
# A view's parent is made before it, so a view of a view is made as a view of the root texture.
for vid, v in m['views'].items():
    seen = set()
    while v['of'] in m['views'] and v['of'] not in seen:
        seen.add(v['of'])
        v['of'] = m['views'][v['of']]['of']
if args.readback == 'last' and written:
    m['readback'] = written[-1]
elif args.readback == 'all':
    m['readback'] = sorted({t for pass_ in written for t in pass_})
with open(args.out, 'w') as f:
    json.dump(m, f, indent=1)
print('%s: %d command buffers, %d buffers, %d textures (+%d views), %d render and %d compute pipelines; readback %s'
      % (args.out, len(m['commandBuffers']), len(m['buffers']), len(m['textures']), len(m['views']),
         len(m['renderPipelines']), len(m['computePipelines']), m.get('readback', [])))
