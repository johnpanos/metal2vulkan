#!/usr/bin/env python3
"""Compare two pipeline-census runs (DIR/census.jsonl each): per state and driver shader, the
instruction counts (instr, non-nops, nops, and the ALU categories cat2/cat3), the SPIR-V words and
the translator's time, then totals over the shaders both runs compiled.

    compare.py BEFORE_DIR AFTER_DIR [--stage FRAG] [--all]

Rows are sorted by the instruction change; --all prints unchanged shaders too. A state's shader
is matched by its stage and its order among the state's shaders of that stage (the program
numbers are the driver's own counter and differ between runs)."""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('before')
ap.add_argument('after')
ap.add_argument('--stage', help='only this driver stage (VERT, BVERT, FRAG, CL)')
ap.add_argument('--all', action='store_true')
args = ap.parse_args()
KEYS = ['instr', 'non_nops', 'nops', 'cat2', 'cat3', 'full', 'max_waves']


def load(d):
    states = {}
    for line in open(os.path.join(d, 'census.jsonl')):
        r = json.loads(line)
        seen = {}
        shaders = {}
        for s in r['shaders']:
            n = seen.get(s['stage'], 0)
            seen[s['stage']] = n + 1
            shaders[(s['stage'], n)] = s['stats']
        r['by'] = shaders
        states[r['id']] = r
    return states


a, b = load(args.before), load(args.after)
rows, totals = [], {k: [0, 0] for k in KEYS}
time = [0, 0]
words = [0, 0]
for sid in sorted(set(a) & set(b)):
    ra, rb = a[sid], b[sid]
    time[0] += ra['cpu_us']
    time[1] += rb['cpu_us']
    words[0] += ra.get('spirv_words') or 0
    words[1] += rb.get('spirv_words') or 0
    for key in sorted(set(ra['by']) & set(rb['by'])):
        if args.stage and key[0] != args.stage:
            continue
        sa, sb = ra['by'][key], rb['by'][key]
        for k in KEYS:
            totals[k][0] += sa.get(k, 0)
            totals[k][1] += sb.get(k, 0)
        d = sb.get('instr', 0) - sa.get('instr', 0)
        if d or args.all:
            rows.append((d, sid, key[0], ra.get('fragment') or ra.get('vertex'), sa, sb))
rows.sort(key=lambda r: (r[0], r[1]))
print('%-6s %-5s %-36s %13s %13s %11s %11s' % ('state', 'stage', 'function', 'instr', 'non-nops', 'cat2', 'cat3'))
for d, sid, stage, name, sa, sb in rows:
    cell = lambda k: '%5d->%-5d' % (sa.get(k, 0), sb.get(k, 0))
    print('%-6s %-5s %-36s %13s %13s %11s %11s' % (sid, stage, (name or '')[:36], cell('instr'), cell('non_nops'),
                                                    cell('cat2'), cell('cat3')))
print('only before: %d states, only after: %d states' % (len(set(a) - set(b)), len(set(b) - set(a))))
for k in KEYS:
    x, y = totals[k]
    print('total %-9s %8d -> %8d  (%+.1f%%)' % (k, x, y, 100.0 * (y - x) / x if x else 0.0))
print('total spirv_words %8d -> %8d' % tuple(words))
print('total cpu_us      %8d -> %8d  (translator and driver, per state creation)' % tuple(time))
