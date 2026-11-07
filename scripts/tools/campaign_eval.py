#!/usr/bin/env python3
"""180 架部署验证 campaign 的验收判定(A1–A4)与探索性补充指标。

验收标准见 results/campaign_180_analysis/acceptance_criteria.md(2026-11-06 20:54 运行中冻结,
逐字继承 ros2_ws_HJH mhe_variant_ab_20261105_181925 预注册第 1/2/9/10 条的 R 部分)。
本脚本只读批次目录;逐架基础判据复用 batch_aggregate.parse_flight(不修改它,驱动在用)。

  A1 无发散            release 段,任何阶段 NMPC solve failed > 100(含 pre-trigger-divergence)
  A2 无持续求解失败     release 段,MHE 're-anchored arrival prior' 次数 = 0
  A3 unhealthy 不进 NMPC 全部合格架次,NMPC 保持区间内模型质量极差 = 0(原口径去转换帧)
  A4 安全回退有效       全部进入 UNRESOLVED 的架次,+5 s 后速度 < 0.5 m/s 且未发散

其余为描述性/探索性(事后补充,不作判定)。二值结局给 Clopper–Pearson 95% 区间。

用法:
  python3 scripts/tools/campaign_eval.py [批次目录 ...] [--out results/campaign_180_analysis]
  不给批次目录时,自动找 batch_info 中 fingerprint=186022e6c95cfa38 且 experiment 属于
  release/faults/thrust/thrust_low_d4 的批次。campaign 未结束时也可运行(报告标注"进行中")。
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_aggregate as ba  # noqa: E402

TAG = 'deployed-r-armd-180-v1'
FP = '186022e6c95cfa38'
SEGMENTS = ('release', 'faults', 'thrust', 'thrust_low_d4')
PLANNED = {'release': 72, 'faults': 48, 'thrust': 48, 'thrust_low_d4': 12}
LIFT_DUR = 8.4
LB_R = 0.70 * ba.M_B
INFRA_NOT_A1 = ('startup-fail', 'logs-missing', 'config-mismatch', 'attach-fail', 'no-trigger')
SOLVE = re.compile(r'\| solve=([0-9.]+)ms')
EZ = re.compile(r'\[model-used\].*? e=\[[-+0-9.]+,[-+0-9.]+,([-+0-9.]+)\]')
AGE = re.compile(r'\[flight-diag\].*?health=(\d) age=([0-9.]+)s')


def cp(k, n, a=0.05):
    if n == 0:
        return (None, None)
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def find_batches(root):
    out = []
    for bi in sorted(glob.glob(os.path.join(root, 'batch_*', 'batch_info.txt'))):
        info = dict(l.rstrip('\n').split('=', 1) for l in open(bi) if '=' in l)
        # 按运行时代码指纹识别:9881e78(仅改论文)在 release 段运行中提交,后三段 git_tag 为空
        if info.get('fingerprint') == FP and info.get('experiment') in SEGMENTS:
            out.append(os.path.dirname(bi))
    return out


def extra(row, base):
    """探索性补充指标 + A2/A3/A4 需要的量。"""
    fdir, st = row['flight_dir'], row['stamp']
    o = {}
    N = ba.wall_lines(os.path.join(fdir, f'grip_nmpc_{st}.log'))
    Mh = ba.wall_lines(os.path.join(fdir, f'grip_mhe_{st}.log'))
    t_end = N[-1][0] if N else None
    t_lift, t_dyn, t_cmd = base.get('t_lift'), base.get('t_dyn'), base.get('t_cmd')
    t_det, t_unres = base.get('t_detach'), base.get('t_unresolved')
    t_trig = t_cmd if t_cmd is not None else base.get('t_inject')
    carry_end = t_trig if t_trig is not None else (t_det if t_det else t_end)
    m_p = base['m_p']
    mt = ba.M_B + m_p

    # A2 / 求解
    o['mhe_reinit'] = sum(1 for _, l in Mh if 're-anchored arrival prior' in l)
    ms = np.asarray([float(x) for _, l in Mh for m in [ba.TIMING.search(l)] if m and m.group(2)
                     for x in m.group(2).split(',')])
    for q in (50, 95, 99):
        o[f'mhe_p{q}_ms'] = float(np.percentile(ms, q)) if len(ms) else None
    o['mhe_max_ms'] = float(ms.max()) if len(ms) else None
    o['mhe_n_solves'] = int(len(ms))
    ns = [float(m.group(1)) for _, l in N for m in [SOLVE.search(l)] if m]
    o['nmpc_solve_med_ms_sampled'] = float(np.median(ns)) if ns else None

    # A3:NMPC 保持区间内模型质量漂移(口径同 aggregate_mhe_variant_ab.mhe_metrics)
    niv = ba.intervals(N, r'\[payload-health\] unhealthy/stale',
                       r'\[payload-health\] healthy/fresh', t_end)
    fr = np.asarray([(w, float(m.group(2)), int(m.group(4))) for w, l in N
                     for m in [ba.MU.search(l)] if m]) if N else np.zeros((0, 3))
    if not len(fr):
        fr = np.zeros((0, 3))
    drift = 0.0
    for a, b in niv:
        s = (fr[:, 0] > a + 0.15) & (fr[:, 0] < b - 0.05)
        if s.sum() > 1:
            drift = max(drift, float(np.ptp(fr[s, 1])))
    o['nmpc_m_drift_while_held'] = drift
    o['nmpc_hold_n_intervals'] = len(niv)
    # 释放后模型清零(原口径:accept+5…20 s 的 model-used m 中位与 m_B 差 < 0.03 且 dropped=1)
    o['model_cleared'] = None
    if base.get('delay_cmd') is not None and t_cmd is not None and len(fr):
        t_acc = t_cmd + base['delay_cmd']
        s = (fr[:, 0] > t_acc + 5.0) & (fr[:, 0] < t_acc + 20.0)
        if s.any():
            o['model_cleared'] = bool(abs(float(np.median(fr[s, 1])) - ba.M_B) < ba.TOL_BAND
                                      and fr[s, 2].min() == 1)

    # A4 / UNRESOLVED 后速度
    o['unres_any'] = t_unres is not None
    o['vmax_after_unres5'] = o['vmax_after_unres'] = None
    fo = ba.one(os.path.join(fdir, 'resid'), 'resid_odom_*.csv')
    fi = ba.one(os.path.join(fdir, 'resid'), 'resid_internal_*.csv')
    if t_unres and fo:
        O = np.genfromtxt(fo, delimiter=',', names=True)
        if O.size > 10:
            tw = O['t_header_ns'] * 1e-9
            spd = np.sqrt(O['vb_x'] ** 2 + O['vb_y'] ** 2 + O['vb_z'] ** 2)
            s0, s5 = tw > t_unres, tw > t_unres + 5.0
            o['vmax_after_unres'] = float(spd[s0].max()) if s0.any() else None
            o['vmax_after_unres5'] = float(spd[s5].max()) if s5.any() else None

    # 质量分段(口径同 fr_exploratory.py)
    for k in ('mae_hover', 'mae_fig8', 'bias_fig8_pct', 'mae_post_10_40', 'collapse_frac',
              'bound_frac', 'unhealthy_frac_loaded'):
        o[k] = None
    if fi and fo:
        it, od = ba.read_csv_cols(fi), ba.read_csv_cols(fo)
        if len(it.get('t_recv_ns', [])) and len(od.get('t_header_ns', [])):
            off = float(np.median(od['t_header_ns'] - od['t_recv_ns'])) * 1e-9
            t = it['t_recv_ns'] * 1e-9 + off
            m = it['m_est']

            def win(a, b):
                if a is None or b is None or b <= a:
                    return np.array([])
                return m[(t >= a) & (t < b) & np.isfinite(m)]
            hov = win(t_lift + LIFT_DUR + 2.0 if t_lift else None, t_dyn)
            f8 = win(t_dyn + 5.0 if t_dyn else None, carry_end)
            post = win(t_det + 10.0 if t_det else None, t_det + 40.0 if t_det else None)
            allm = win(t_lift, carry_end)
            if len(hov) > 5:
                o['mae_hover'] = float(np.mean(np.abs(hov - mt)))
            if len(f8) > 5:
                o['mae_fig8'] = float(np.mean(np.abs(f8 - mt)))
                o['bias_fig8_pct'] = float(100 * np.median(f8 - mt) / mt)
                o['collapse_frac'] = float(np.mean((f8 - ba.M_B) < 0.5 * m_p))
            if len(post) > 5:
                o['mae_post_10_40'] = float(np.mean(np.abs(post - ba.M_B)))
            if len(allm) > 5:
                o['bound_frac'] = float(np.mean(allm <= LB_R + 0.002))
    ul = base.get('health_unhealthy_loaded_s')
    if ul is not None and t_lift and carry_end and carry_end > t_lift:
        o['unhealthy_frac_loaded'] = float(ul / (carry_end - t_lift))

    # 跟踪:竖直平均误差(带载 8 字)
    ez = [float(m.group(1)) for w, l in N for m in [EZ.search(l)]
          if m and t_dyn and carry_end and t_dyn + 5 < w < carry_end]
    o['ez_mean_fig8_mm'] = 1e3 * float(np.mean(ez)) if ez else None

    # 执行器 / 内环(flight-diag 5 s 窗):带载与投放后
    t_ref = t_det if t_det is not None else (base.get('t_lost') or t_trig)
    for tag, a, b in (('loaded', t_dyn + 5 if t_dyn else None, carry_end),
                      ('post', t_ref, t_ref + ba.POST_WIN if t_ref else None)):
        fd, om, age = [], [], []
        for w, l in N:
            if a is None or b is None or not (a < w < b):
                continue
            m = ba.FD.search(l)
            if m:
                fd.append([float(x) for x in m.groups()])
            m = ba.OM.search(l)
            if m:
                om.append(float(m.group(1)))
            m = AGE.search(l)
            if m:
                age.append(float(m.group(2)))
        fd = np.asarray(fd)
        for j, nm in ((1, 'satT'), (2, 'satR'), (3, 'satP'), (4, 'satY'), (6, 'motsat')):
            o[f'{nm}_{tag}_pct'] = float(fd[:, j].mean()) if len(fd) else None
        o[f'om_scale_min_{tag}'] = float(min(om)) if om else None
        o[f'payload_age_max_{tag}_s'] = float(max(age)) if age else None

    # 释放检测:watchdog 武装时刻(相对抬升)
    ta = base.get('t_armed')
    o['armed_after_lift_s'] = (ta - t_lift) if (ta and t_lift) else None
    return o


def load(bdirs):
    F = []
    for bd in bdirs:
        info = dict(l.rstrip('\n').split('=', 1) for l in open(os.path.join(bd, 'batch_info.txt')) if '=' in l)
        seg = info['experiment']
        for r in csv.DictReader(open(os.path.join(bd, 'manifest.csv'))):
            b = ba.parse_flight(r)
            b['segment'] = seg
            b['batch'] = os.path.basename(bd)
            b['fingerprint_row'] = r.get('fingerprint', '')
            if r.get('stamp') not in ('', 'NA', None):
                b.update(extra(r, b))
            F.append(b)
    return F


def a1_set(F):
    return [f for f in F if f['segment'] == 'release' and
            (f.get('eligible') or f.get('ineligible_reason', '').startswith('pre-trigger-divergence'))]


def acceptance(F):
    S1 = a1_set(F)
    E = [f for f in F if f.get('eligible')]
    U = [f for f in E if f.get('unres_any')]
    res = {}
    k = sum(1 for f in S1 if f.get('diverged'))
    res['A1'] = dict(rule='release: no divergence (any phase)', n=len(S1), events=k,
                     ids=[f['idx'] for f in S1 if f.get('diverged')], ok=(k == 0))
    k = sum(1 for f in S1 if (f.get('mhe_reinit') or 0) > 0)
    res['A2'] = dict(rule='release: no MHE re-initialisation', n=len(S1), events=k,
                     total_reinit=sum(f.get('mhe_reinit') or 0 for f in S1),
                     ids=[f['idx'] for f in S1 if (f.get('mhe_reinit') or 0) > 0], ok=(k == 0))
    bad = [f for f in E if (f.get('nmpc_m_drift_while_held') or 0) != 0]
    res['A3'] = dict(rule='all eligible: NMPC model mass constant while held', n=len(E),
                     events=len(bad), ids=[(f['segment'], f['idx']) for f in bad], ok=not bad)
    badu = [f for f in U if not (f.get('hover_ok') is True and not f.get('diverged'))]
    res['A4'] = dict(rule='every UNRESOLVED: v<0.5 m/s after +5 s and no divergence', n=len(U),
                     events=len(badu), ids=[(f['segment'], f['idx']) for f in badu],
                     ok=(None if not U else not badu))
    return res


BIN = [('attached', '物理吸附'), ('complete', '确认完成'), ('early_confirm', '提前确认'),
       ('missed', '漏检'), ('unresolved', 'UNRESOLVED'), ('diverged', '发散'),
       ('carry_false_confirm', '带载误确认'), ('lost_detected', 'LOST 检出'),
       ('jam_wrong_confirm', '卡死误确认'), ('model_cleared', '释放后模型清零')]
CONT = [('delay_cmd', '确认延迟(指令后) s', '{:.2f}'), ('delay_det', '确认延迟(脱离后) s', '{:.2f}'),
        ('mass_loaded_mae', '带载质量 MAE kg', '{:.4f}'), ('mae_hover', '带载悬停质量 MAE kg', '{:.4f}'),
        ('mae_fig8', '带载 8 字质量 MAE kg', '{:.4f}'), ('mass_post_mae', '投放后质量 MAE kg', '{:.4f}'),
        ('collapse_frac', 'collapse fraction', '{:.3f}'), ('bound_frac', '贴下界比例', '{:.4f}'),
        ('attach_conv_s', '抬升后收敛 s', '{:.2f}'), ('detach_conv_s', '回到空载 s', '{:.2f}'),
        ('unhealthy_frac_loaded', '带载不健康时间占比', '{:.3f}'),
        ('rmse_loaded', '带载 RMSE m', '{:.3f}'), ('rmse_post', '投放后 RMSE m', '{:.3f}'),
        ('peak_post', '投放后峰值误差 m', '{:.3f}'), ('ez_mean_fig8_mm', '8 字平均竖直误差 mm', '{:.1f}'),
        ('mhe_p50_ms', 'MHE P50 ms', '{:.3f}'), ('mhe_p95_ms', 'MHE P95 ms', '{:.3f}'),
        ('mhe_p99_ms', 'MHE P99 ms', '{:.3f}'), ('mhe_max_ms', 'MHE max ms', '{:.3f}'),
        ('nmpc_solve_med_ms_sampled', 'NMPC 求解中位(抽样) ms', '{:.2f}'),
        ('mhe_nonzero_status', 'MHE 非零状态数', '{:.0f}'), ('mhe_reinit', 'MHE 重初始化数', '{:.0f}'),
        ('satT_loaded_pct', '带载推力饱和 %', '{:.2f}'), ('satR_loaded_pct', '带载 roll 力矩饱和 %', '{:.2f}'),
        ('satP_loaded_pct', '带载 pitch 力矩饱和 %', '{:.2f}'), ('satY_loaded_pct', '带载 yaw 力矩饱和 %', '{:.2f}'),
        ('om_scale_min_loaded', '带载 om_scale 最小', '{:.3f}'),
        ('payload_age_max_loaded_s', '带载载荷帧解龄最大 s', '{:.3f}'),
        ('armed_after_lift_s', 'watchdog 武装(抬升后) s', '{:.1f}'),
        ('vmax_after_unres', 'UNRESOLVED 后最大速度 m/s', '{:.2f}')]


def summ(vals, f):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return '—'
    return f"{f.format(np.median(v))} [{f.format(min(v))}, {f.format(max(v))}] (n={len(v)})"


def report(F, bdirs, out):
    os.makedirs(out, exist_ok=True)
    acc = acceptance(F)
    flown = {s: sum(1 for f in F if f['segment'] == s and f.get('kind') != 'rerun') for s in SEGMENTS}
    done = all(flown[s] >= PLANNED[s] for s in SEGMENTS)
    fps = sorted({f['fingerprint_row'] for f in F})
    L = [f"# 180 架部署验证 campaign:验收与描述性结果{'' if done else '(进行中,非最终)'}", '',
         '- 批次:' + ', '.join(os.path.basename(b) for b in bdirs),
         '- 已飞(不含补跑)/计划:' + ', '.join(f"{s} {flown[s]}/{PLANNED[s]}" for s in SEGMENTS),
         f"- manifest 指纹:{', '.join(fps)}" + ('' if fps == [FP] else f"  ⚠️ 期望仅 {FP}"),
         '- 不合格:' + (', '.join(f"{f['segment']}#{f['idx']} {f['ineligible_reason']}"
                                for f in F if not f.get('eligible')) or '无'),
         '', '## 验收(acceptance_criteria.md,继承 11-05 预注册第 1/2/9/10 条)', '',
         '| 判据 | 内容 | n | 事件 | 结论 |', '|---|---|---:|---:|---|']
    for k, v in acc.items():
        verdict = '未被检验' if v['ok'] is None else ('满足' if v['ok'] else f"不满足 {v['ids']}")
        L.append(f"| {k} | {v['rule']} | {v['n']} | {v['events']} | {verdict} |")
    oks = [v['ok'] for v in acc.values()]
    L += ['', f"总体:{'全部满足' if all(o is not False for o in oks) else '未全部满足'}"
          f"{'(A4 未被检验)' if acc['A4']['ok'] is None else ''}"
          f"{'' if done else ' —— campaign 未结束,仅为中期核对'}", '']
    L += ['## 二值结局(合格架次;k/n 与 Clopper–Pearson 95%)', '']
    groups = {}
    for f in F:
        if f.get('eligible'):
            groups.setdefault((f['segment'], f['fault'], f['tgain'], f['scen']), []).append(f)
    L += ['| 段 | 故障 | 增益 | 脱离 | n | ' + ' | '.join(n for _, n in BIN) + ' |',
          '|' + '---|' * (5 + len(BIN))]
    for g in sorted(groups):
        G = groups[g]
        cells = []
        for key, _ in BIN:
            vv = [f.get(key) for f in G if f.get(key) is not None]
            k = sum(1 for x in vv if x)
            lo, hi = cp(k, len(vv))
            cells.append('—' if not vv else f"{k}/{len(vv)} [{lo:.2f},{hi:.2f}]")
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {len(G)} | " + ' | '.join(cells) + ' |')
    L += ['', '## 连续量(合格架次;中位 [最小, 最大];除注明外为事后补充的探索性指标)', '']
    for g in sorted(groups):
        G = groups[g]
        L += [f"### {g[0]} / {g[1]} / k={g[2]} / {g[3]}(n={len(G)})", '', '| 指标 | 值 |', '|---|---|']
        for key, nm, fm in CONT:
            s = summ([f.get(key) for f in G], fm)
            if s != '—':
                L.append(f"| {nm} | {s} |")
        L.append('')
    L += ['## 说明', '',
          '- SITL 重复之间没有可播种的随机输入;架次间差异来自仿真步进与多进程调度的时序抖动。'
          '区间描述的是这台机器上本仿真栈的重复变异,不是对真实飞行总体的推断。',
          '- 本报告不做显著性检验,因此不涉及多重比较校正。',
          '- NMPC 求解时间只在 [model-used] 行抽样记录(0.1 ms 分辨率),只给中位数;'
          'P99、最大值和截止时间超时次数无法从本批日志得到。', '']
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L))
    json.dump({'done': done, 'flown': flown, 'fingerprints': fps, 'acceptance': acc},
              open(os.path.join(out, 'acceptance.json'), 'w'), indent=1, default=str)
    keys = sorted({k for f in F for k in f})
    with open(os.path.join(out, 'per_flight.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for f in sorted(F, key=lambda x: (SEGMENTS.index(x['segment']), x['idx'])):
            w.writerow(f)
    return acc, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bdirs', nargs='*')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ws = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bdirs = a.bdirs or find_batches(os.path.join(ws, 'results'))
    out = a.out or os.path.join(ws, 'results', 'campaign_180_analysis')
    F = load(bdirs)
    acc, done = report(F, bdirs, out)
    for k, v in acc.items():
        print(k, v['n'], v['events'], v['ok'])
    print('done' if done else 'in-progress', '->', out)


if __name__ == '__main__':
    sys.exit(main())
