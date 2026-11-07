#!/usr/bin/env python3
"""Summarise review_replay_ablation.py output. Usage: python3 review_replay_summary.py <rev_results_R.json>"""
import json, sys, numpy as np
R = json.load(open(sys.argv[1]))
rng = np.random.default_rng(0)


def ci(d):
    bs = np.median(rng.choice(d, (10000, len(d))), axis=1)
    return np.percentile(bs, [2.5, 97.5])


print('replay-vs-online loaded |diff| kg: median %.2e max %.2e' %
      (np.median([r['replay_vs_online_mae'] for r in R]), max(r['replay_vs_online_mae'] for r in R)))
mets = ['mae_loaded', 'mae_fig8', 'bias_fig8_pct', 'mae_post', 'conv_lift', 'conv_det']
ests = ['online', 'R_base', 'R_com', 'R_term', 'R_com_term', 'WFB', 'LPF', 'RLS', 'EKF']
for grp, f in [('all', lambda r: True), ('4m/s', lambda r: r['v'] == 4.0), ('2m/s', lambda r: r['v'] == 2.0)]:
    S = [r for r in R if f(r)]
    print(f'\n== {grp} n={len(S)}  (medians; conv: n finite)')
    print('est'.ljust(11) + ''.join(m.rjust(15) for m in mets))
    for e in ests:
        row = e.ljust(11)
        for m in mets:
            x = np.array([r[f'{e}.{m}'] for r in S], float)
            fin = np.isfinite(x)
            row += (f'{np.median(x[fin]):.4f}({fin.sum()})' if fin.any() else 'nan').rjust(15)
        print(row)
    print('-- paired vs R_base: median delta [95% CI] better/same/worse')
    for e in ['R_com', 'R_term', 'R_com_term', 'WFB', 'LPF', 'RLS', 'EKF']:
        for m in ['mae_loaded', 'mae_fig8', 'mae_post', 'conv_lift', 'conv_det']:
            a = np.array([r[f'R_base.{m}'] for r in S], float); b = np.array([r[f'{e}.{m}'] for r in S], float)
            ok = np.isfinite(a) & np.isfinite(b)
            nb = int((~np.isfinite(b) & np.isfinite(a)).sum())
            if ok.sum() < 3:
                continue
            d = b[ok] - a[ok]; lo, hi = ci(d); tol = 1e-5
            print(f'  {e:10s} {m:11s} d={np.median(d):+.5f} [{lo:+.5f},{hi:+.5f}] '
                  f'{(d < -tol).sum()}/{(abs(d) <= tol).sum()}/{(d > tol).sum()}  (est censored where base not: {nb})')
tr = [r for r in R if 'tr.dv_rms' in r]
for v in (2.0, 4.0):
    S = [r for r in tr if r['v'] == v]
    for k in ['dv_rms', 'dv_max', 'acc_rms', 'acc_z_mean', 'w_rms', 'alpha_rms']:
        x = np.array([r[f'tr.{k}'] for r in S])
        print(f'transport v={v} {k}: median {np.median(x):.4g} max {np.max(x):.4g}')
