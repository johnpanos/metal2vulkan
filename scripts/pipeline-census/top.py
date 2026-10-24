#!/usr/bin/env python3
"""The census's heaviest driver shaders: the top N by IR3 instruction count of one stage, each
with its category split and the opcodes that dominate it, as a Markdown table.

    top.py CENSUS_DIR [--stage FRAG] [-n 10]

Opcodes come from each row's kept log (pipeline-census --keep-logs): the disassembly lines
before the shader's statistics, counted by mnemonic (`add.f`, `sel.b32`, `cov.u16f32`, ...),
prefixes such as (sy) and (rpt) dropped. A shader made by several states or draws is listed
once, at its first row."""
import argparse
import collections
import json
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('census')
ap.add_argument('--stage', default='FRAG')
ap.add_argument('-n', type=int, default=10)
args = ap.parse_args()
OP = re.compile(r'^(?:MESA: info: )?\s*(?:\((?:sy|ss|jp|eq|ul|nop\d*|rpt\d*|sat|ei)\)\s*)*([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\b')
STATS = re.compile(r'; (\w+) prog (\S+): ')
SKIP = {'end', 'nop', 'shader', 'name', 'internal', 'inputs', 'outputs', 'decl_var', 'decl_function', 'impl', 'block',
        'max_subgroup_size', 'min_subgroup_size', 'info', 'num_ubos', 'num_driver_params', 'offsets', 'ubo_state'}


def opcodes(log, stage, program):
    """The mnemonics of the disassembly before `; STAGE prog PROGRAM:`, in that log."""
    if not os.path.exists(log):
        return None
    lines = open(log, errors='replace').read().splitlines()
    end = next((i for i, l in enumerate(lines) if '; %s prog %s:' % (stage, program) in l), None)
    if end is None:
        return None
    count = collections.Counter()
    for l in reversed(lines[:end]):
        body = l[len('MESA: info: '):] if l.startswith('MESA: info: ') else l
        if body.startswith('; '):
            continue
        if STATS.search(l) or 'shader: MESA_SHADER' in l or body.startswith('impl ') or '%' in body:
            break
        m = OP.match(l)
        if m and m.group(1).split('.')[0] not in SKIP:
            count[m.group(1)] += 1
    return count


rows, seen = [], set()
for line in open(os.path.join(args.census, 'census.jsonl')):
    r = json.loads(line)
    for s in r.get('shaders', []):
        if s['stage'] != args.stage:
            continue
        key = (r.get('fragment') or r.get('vertex'), json.dumps(s['stats'], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        rows.append((s['stats'].get('instr', 0), r, s))
rows.sort(key=lambda x: -x[0])
print('| # | row | function | instr | non-nops | nops | cat2 ALU | cat3 MAD | cat5 tex | cat6 mem | regs | waves | dominant opcodes |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, (instr, r, s) in enumerate(rows[:args.n], 1):
    st = s['stats']
    ops = opcodes(os.path.join(args.census, 'logs', r['id'] + '.log'), s['stage'], s['program'])
    top = ', '.join('%s %d' % (o, c) for o, c in ops.most_common(5)) if ops else '(run with --keep-logs)'
    name = r.get('fragment') if args.stage == 'FRAG' else r.get('vertex') or r.get('fragment')
    print('| %d | %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %s |' % (
        k, r['id'], name, instr, st.get('non_nops', 0), st.get('nops', 0), st.get('cat2', 0), st.get('cat3', 0),
        st.get('cat5', 0), st.get('cat6', 0), st.get('full', 0), st.get('max_waves', 0), top))
total = sum(x[0] for x in rows)
print('\n%d distinct %s shaders, %d instructions in all' % (len(rows), args.stage, total))
