#!/usr/bin/env python3
"""F/R 闭环配对批次(mhe_variant_ab)的**探索性**补充指标(不在预注册内)。

预注册指标见批次目录的 preregistration.md / summary.md;本脚本补算的量只作探索性报告:
  - 质量误差分段:带载悬停 / 带载 8 字 / 投放后(口径同论文第 5 章:悬停 = 抬升斜坡
    结束后 2 s 到 8 字开始;8 字 = 开始后 5 s 到释放指令;投放后 = 脱离后 10–40 s)
  - 塌陷比例:8 字中 m̂_P < 0.5·m_P(真值)的样本占比
  - 贴边界比例:m̂ 距质量下界 2 g 以内的样本占比(F: 0.95 m_B;R: 0.70 m_B)
  - 竖直平均误差:带载 8 字 [model-used] e_z 的均值
  - MHE 求解耗时 P99 / 最大值([mhe-timing] 原始序列);NMPC 求解耗时(5 s 摘要行)
统计:配对单位 pair_id,差值 R−F(偏差类 |R|−|F|),中位数、bootstrap 95% CI(10k,
种子 0)、Wilcoxon 符号秩(双侧)、R 优/同/劣。
用法: python3 fr_exploratory.py <批次 main 目录> [--lift-dur 8.4]
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

M_B = 2.0643
WALL = re.compile(r'^\[\w+\] \[(\d+\.\d+)\]')
TIMING = re.compile(r'\[mhe-timing\] variant=\w+ ms=\[([0-9.,]*)\]')
EZ = re.compile(r'\[model-used\].*? e=\[[-+0-9.]+,[-+0-9.]+,([-+0-9.]+)\]')
NSOLVE = re.compile(r'\| solve=([0-9.]+)ms')


def read_cols(path):
    with open(path) as f:
        head = f.readline().strip().split(',')
        rows = []
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) == len(head):
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    a = np.asarray(rows) if rows else np.empty((0, len(head)))
    return {h: a[:, i] for i, h in enumerate(head)}


def one(d, pat):
    g = sorted(glob.glob(os.path.join(d, pat)))
    return g[-1] if g else None


def wall_lines(path):
    out = []
    if path and os.path.exists(path):
        for line in open(path, errors='replace'):
            m = WALL.match(line)
            if m:
                out.append((float(m.group(1)), line))
    return out


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def flight_metrics(row, fdir, lift_dur):
    t_lift, t_dyn = fnum(row['t_lift']), fnum(row['t_dyn'])
    t_cmd, t_det = fnum(row['t_cmd']), fnum(row['t_detach'])
    m_p = float(row['m_p'])
    mt = M_B + m_p
    lb = 0.95 * M_B if row['mhe'] == 'F' else 0.70 * M_B
    out = {}
    rd = os.path.join(fdir, 'resid')
    fi, fo = one(rd, 'resid_internal_*.csv'), one(rd, 'resid_odom_*.csv')
    if fi and fo:
        it, od = read_cols(fi), read_cols(fo)
        off = float(np.median(od['t_header_ns'] - od['t_recv_ns'])) * 1e-9
        t = it['t_recv_ns'] * 1e-9 + off
        m = it['m_est']

        def win(a, b):
            if a is None or b is None or b <= a:
                return np.array([])
            s = (t >= a) & (t < b) & np.isfinite(m)
            return m[s]
        hov = win(t_lift + lift_dur + 2.0 if t_lift else None, t_dyn)
        f8 = win(t_dyn + 5.0 if t_dyn else None, t_cmd)
        post = win(t_det + 10.0 if t_det else None, t_det + 40.0 if t_det else None)
        out['mae_hover'] = float(np.mean(np.abs(hov - mt))) if len(hov) > 5 else None
        out['mae_fig8'] = float(np.mean(np.abs(f8 - mt))) if len(f8) > 5 else None
        out['bias_fig8_pct'] = float(100 * np.median(f8 - mt) / mt) if len(f8) > 5 else None
        out['mae_post_10_40'] = float(np.mean(np.abs(post - M_B))) if len(post) > 5 else None
        out['collapse_frac'] = (float(np.mean((f8 - M_B) < 0.5 * m_p))
                                if len(f8) > 5 else None)
        allm = win(t_lift, t_cmd)
        out['bound_frac'] = (float(np.mean(allm <= lb + 0.002)) if len(allm) > 5 else None)
    st = row['stamp']
    Mh = wall_lines(os.path.join(fdir, f'grip_mhe_{st}.log'))
    ms = np.asarray([float(x) for _, l in Mh for mm in [TIMING.search(l)] if mm and mm.group(1)
                     for x in mm.group(1).split(',')])
    out['mhe_p99_ms'] = float(np.percentile(ms, 99)) if len(ms) else None
    out['mhe_max_ms'] = float(ms.max()) if len(ms) else None
    N = wall_lines(os.path.join(fdir, f'grip_nmpc_{st}.log'))
    ez = [float(mm.group(1)) for w, l in N for mm in [EZ.search(l)]
          if mm and t_dyn and t_cmd and t_dyn + 5 < w < t_cmd]
    out['ez_mean_fig8_mm'] = 1e3 * float(np.mean(ez)) if ez else None
    ns = [float(mm.group(1)) for _, l in N for mm in [NSOLVE.search(l)] if mm]
    out['nmpc_solve_med_ms'] = float(np.median(ns)) if ns else None
    return out


def paired(rows, key, absdiff=False, tol=1e-9):
    d = []
    for f, r in rows:
        a, b = f.get(key), r.get(key)
        if a is None or b is None:
            continue
        d.append((abs(b) - abs(a)) if absdiff else (b - a))
    d = np.asarray(d)
    if not len(d):
        return None
    rng = np.random.default_rng(0)
    bs = np.median(rng.choice(d, (10000, len(d))), axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try:
        p = wilcoxon(d).pvalue if np.any(np.abs(d) > tol) else 1.0
    except ValueError:
        p = 1.0
    better = int(np.sum(d < -tol)); worse = int(np.sum(d > tol))
    fm = np.median([f[key] for f, r in rows if f.get(key) is not None and r.get(key) is not None])
    rm = np.median([r[key] for f, r in rows if f.get(key) is not None and r.get(key) is not None])
    return len(d), fm, rm, float(np.median(d)), lo, hi, p, better, len(d) - better - worse, worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bdir')
    ap.add_argument('--lift-dur', type=float, default=8.4)
    a = ap.parse_args()
    man = {r['idx']: r for r in csv.DictReader(open(os.path.join(a.bdir, 'manifest.csv')))}
    pf = list(csv.DictReader(open(os.path.join(a.bdir, 'per_flight.csv'))))
    by_pair = {}
    for row in pf:
        if row.get('eligible') != 'True':
            continue
        m = flight_metrics(row, man[row['idx']]['flight_dir'], a.lift_dur)
        m.update(work=row['work'], scen=row['scen'], m_p=float(row['m_p']), v=float(row['v']))
        by_pair.setdefault(row['pair_id'], {})[row['mhe']] = m
    pairs = [(p['F'], p['R']) for p in by_pair.values() if 'F' in p and 'R' in p]
    keys = [('mae_hover', False), ('mae_fig8', False), ('bias_fig8_pct', True),
            ('collapse_frac', False), ('bound_frac', False), ('mae_post_10_40', False),
            ('ez_mean_fig8_mm', True), ('mhe_p99_ms', False), ('mhe_max_ms', False),
            ('nmpc_solve_med_ms', False)]
    groups = [('all', lambda f: True),
              ('m_p=0.15', lambda f: f['m_p'] == 0.15), ('m_p=0.30', lambda f: f['m_p'] == 0.30),
              ('v=2', lambda f: f['v'] == 2.0), ('v=4', lambda f: f['v'] == 4.0),
              ('d0', lambda f: f['scen'] == 'd0'), ('d4', lambda f: f['scen'] == 'd4')]
    print(f'# 探索性补充指标(非预注册)— {len(pairs)} 对\n')
    print('差值 R−F(偏差类 |R|−|F|),负 = R 更好\n')
    for gname, g in groups:
        P = [(f, r) for f, r in pairs if g(f)]
        print(f'## {gname}(n={len(P)} 对)\n')
        print('| 指标 | n | F 中位 | R 中位 | 差中位 | 95% CI | Wilcoxon p | R优/同/劣 |')
        print('|---|---:|---:|---:|---:|---|---:|---|')
        for k, ab in keys:
            res = paired(P, k, ab)
            if res is None:
                continue
            n, fm, rm, dm, lo, hi, p, b, s, w = res
            print(f'| {k} | {n} | {fm:.4g} | {rm:.4g} | {dm:+.4g} | [{lo:+.4g}, {hi:+.4g}] '
                  f'| {p:.3g} | {b}/{s}/{w} |')
        print()
    print('## 逐格原始值(每格 6 对:F → R)\n')
    for w in ('c1', 'c2', 'c3', 'c4'):
        for sc in ('d0', 'd4'):
            P = [(f, r) for f, r in pairs if f['work'] == w and f['scen'] == sc]
            fv = ', '.join(f"{f['mae_fig8']:.4f}" for f, _ in P if f.get('mae_fig8') is not None)
            rv = ', '.join(f"{r['mae_fig8']:.4f}" for _, r in P if r.get('mae_fig8') is not None)
            print(f'- {w}/{sc} 8 字质量 MAE [kg]  F: {fv}  |  R: {rv}')


if __name__ == '__main__':
    sys.exit(main())
