#!/usr/bin/env python3
"""比对两次 harness 输出。发布输出要求逐位相同;运行期日志逐条比对并列出差异。
用法: python3 nmpc_equiv_compare.py ref.pkl new.pkl [--logs]"""
import difflib
import pickle
import sys

IGNORE_TOPICS = set()
IGNORE_CALLS = ('/mavros/param/',)

a = pickle.load(open(sys.argv[1], 'rb'))
b = pickle.load(open(sys.argv[2], 'rb'))
show_logs = '--logs' in sys.argv

pa = [r for r in a['pubs'] if r[1] not in IGNORE_TOPICS]
pb = [r for r in b['pubs'] if r[1] not in IGNORE_TOPICS]
ok = True
if pa != pb:
    ok = False
    n = min(len(pa), len(pb))
    i = next((i for i in range(n) if pa[i] != pb[i]), n)
    print(f'PUBS DIFFER: len {len(pa)} vs {len(pb)}, first diff at #{i}')
    for r in (pa[i:i+1], pb[i:i+1]):
        print('   ', r[0][:2] if r else None, (r[0][2][:600] if r else ''))
else:
    print(f'pubs identical ({len(pa)})')

ca = [c for c in a['calls'] if not c[1].startswith(IGNORE_CALLS)]
cb = [c for c in b['calls'] if not c[1].startswith(IGNORE_CALLS)]
if ca != cb:
    ok = False
    print(f'CALLS DIFFER: {len(ca)} vs {len(cb)}')
else:
    print(f'service calls identical ({len(ca)})')

import re
WALL = re.compile(r"wall=\S+")
la = [WALL.sub("wall=*", f"{lv} {m}") for _, lv, m in a["logs"][a["init_log_count"]:]]
lb = [WALL.sub("wall=*", f"{lv} {m}") for _, lv, m in b["logs"][b["init_log_count"]:]]
if la != lb:
    d = [l for l in difflib.unified_diff(la, lb, lineterm='', n=0)
         if not l.startswith(('---', '+++', '@@'))]
    print(f'runtime logs differ: {len(d)} lines')
    for l in (d if show_logs else d[:20]):
        print('   ', l[:240])
else:
    print(f'runtime logs identical ({len(la)})')
print('EVENTS', a['events'], b['events'])
print('RESULT', 'EQUIVALENT' if ok else 'NOT EQUIVALENT')
sys.exit(0 if ok else 1)
