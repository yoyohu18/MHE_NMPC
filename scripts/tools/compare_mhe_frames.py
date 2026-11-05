#!/usr/bin/env python3
"""比对 replay_mhe_node.py 产生的两份发布记录(旧节点 vs 新节点)。

NMPC 消费的字段必须逐帧一致:m_total、dJ_diag(3)、J_diag(3)、空载置信度、健康位、解龄。
s_xy / c_xy 单独报告(新节点恒为 0;主线 NMPC 偏心补偿关闭,不消费它们)。
用法: python3 compare_mhe_frames.py old.npz new.npz
"""
import sys

import numpy as np

CONSUMED = {'m_total': [0], 'dJ_diag': [5, 6, 7], 'J_diag': [8, 9, 10],
            'confidence': [11], 'healthy': [12], 'solution_age': [13]}
IGNORED = {'s_xy': [1, 2], 'c_xy': [3, 4]}


def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    ok = True
    if a['pe'].shape != b['pe'].shape or not np.array_equal(a['pe_t'], b['pe_t']):
        print(f'FRAME COUNT/TIMING DIFFER: {a["pe"].shape} vs {b["pe"].shape}')
        return 1
    print(f'frames: {len(a["pe"])}')
    for name, idx in CONSUMED.items():
        d = np.max(np.abs(a['pe'][:, idx] - b['pe'][:, idx]))
        flag = 'OK ' if d <= 1e-12 else 'DIFF'
        ok &= d <= 1e-12
        print(f'  {flag} {name:13s} max|old-new| = {d:.3e}')
    for name, idx in IGNORED.items():
        d = np.max(np.abs(a['pe'][:, idx] - b['pe'][:, idx]))
        print(f'  --  {name:13s} max|old-new| = {d:.3e}  (not consumed by mainline NMPC)')
    same_mp = (len(a['mp']) == len(b['mp']) and np.array_equal(a['mp_t'], b['mp_t'])
               and np.max(np.abs(a['mp'] - b['mp']), initial=0.0) <= 1e-12)
    print(f'  {"OK " if same_mp else "DIFF"} mass topic: {len(a["mp"])} vs {len(b["mp"])} msgs')
    ok &= same_mp
    h = a['pe'][:, 12]
    print(f'  healthy fraction {h.mean():.3f}; m_total range '
          f'[{a["pe"][:, 0].min():.4f}, {a["pe"][:, 0].max():.4f}]')
    print('EQUIVALENT' if ok else 'NOT EQUIVALENT')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
