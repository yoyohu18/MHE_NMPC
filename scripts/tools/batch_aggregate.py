#!/usr/bin/env python3
"""主线批次(scripts/run_batch.sh)的逐架指标与按格汇总。

用法:
  python3 batch_aggregate.py <批次目录>                      # per_flight.csv / summary.md / summary.json
  python3 batch_aggregate.py <批次目录> --quiet
  python3 batch_aggregate.py <批次目录> --emit-infra-reruns  # 追加基础设施补跑行,末行打印行数

逐架判据移植自 ros2_ws_HJH 的 aggregate_release_paper.parse_flight(312 架释放实验)与
aggregate_mhe_variant_ab.mhe_metrics(F/R 配对实验),口径不变:事件时刻一律取各节点
日志的墙钟;发散 = solve failed > 100;投放后窗口 = 物理脱离(或注入)后 25 s。
只读本目录的 schedule/manifest 与 flights/ 下的日志。

资格(跑前冻结):只排除基础设施失败 —— 栈未起/无 NMPC 日志、日志缺失、配置与排程
不符、物理吸附失败,以及触发前发散/超时。漏检、提前确认、UNRESOLVED、求解失败、
投放后发散都是结果,不排除。
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

WORK = {'c1': (0.15, 2.0), 'c2': (0.15, 4.0), 'c3': (0.30, 4.0), 'c4': (0.30, 2.0)}
WALL = re.compile(r'^\[\w+\] \[(\d+\.\d+)\]')
EVT = re.compile(r'\[EVT\] name=(\w+) wall=([\d.]+)(.*)')
M_B = 2.0643
DIV_SF = 100
POST_WIN = 25.0
TOL_BAND = 0.03
HOLD = 10
INFRA = ('startup-fail', 'logs-missing', 'config-mismatch', 'attach-fail',
         'no-trigger', 'pre-trigger-divergence')
MU = re.compile(r'\[model-used\].*?pos_err=([0-9.]+) m=([0-9.]+) g0=([-0-9.e]+).*?dropped=(\d)')
AE = re.compile(r'att_err=([0-9.]+|nan)')
FD = re.compile(r'tilt_max=([0-9.]+)deg.*u_sat=\[T([0-9.]+) r([0-9.]+) p([0-9.]+) '
                r'y([0-9.]+)\]% \| mot_peak=([0-9.]+) mot_sat=([0-9.]+)%')
AC = re.compile(r'\[acc-det\] m_acc=([0-9.na]+) acc_conf=([0-9.]+) fresh=(\d)')
OM = re.compile(r'\[flight-diag\].*?om_scale=([0-9.]+)')
TIMING = re.compile(r'\[mhe-timing\] variant=(\w+) ms=\[([0-9.,]*)\]')
FAIL = re.compile(r'MHE solve failed \[status=(-?\d+)\]')


# ---------------------------------------------------------------- 解析工具
def wall_lines(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, errors='replace'):
        m = WALL.match(line)
        if m:
            out.append((float(m.group(1)), line))
    return out


def evts(lines):
    out = []
    for _, l in lines:
        m = EVT.search(l)
        if m:
            out.append((m.group(1), float(m.group(2)), dict(re.findall(r'(\w+)=([^\s]+)', m.group(3)))))
    return out


def first(lines, pat, after=None):
    for w, l in lines:
        if (after is None or w >= after) and re.search(pat, l):
            return w
    return None


def one(d, pat):
    g = sorted(glob.glob(os.path.join(d, pat)))
    return g[-1] if g else None


def read_env(path):
    d = {}
    if path and os.path.exists(path):
        for l in open(path, errors='replace'):
            if '=' in l:
                k, v = l.rstrip('\n').split('=', 1)
                d[k.strip()] = v.strip()
    return d


def rms(v):
    return float(math.sqrt(np.mean(np.square(v)))) if len(v) else None


def read_csv_cols(path):
    with open(path) as f:
        head = f.readline().strip().split(',')
        rows = []
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) != len(head):
                continue
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                continue
    a = np.asarray(rows, dtype=float) if rows else np.empty((0, len(head)))
    return {h: a[:, i] for i, h in enumerate(head)}


def stable_delay(t, m, event, target):
    if event is None or not len(t):
        return None
    for j in np.flatnonzero(t >= event):
        if j + HOLD <= len(m) and np.all(np.abs(m[j:j + HOLD] - target) <= TOL_BAND):
            return float(t[j] - event)
    return None


def seg_err(t, m, a, b, target):
    if a is None or b is None or b <= a:
        return None, None
    s = (t >= a) & (t < b) & np.isfinite(m)
    if s.sum() < 5:
        return None, None
    e = m[s] - target
    return float(100 * np.median(e) / target), float(np.mean(np.abs(e)))


def intervals(lines, on_pat, off_pat, t_end):
    out, cur = [], None
    for w, l in lines:
        if cur is None and re.search(on_pat, l):
            cur = w
        elif cur is not None and re.search(off_pat, l):
            out.append((cur, w))
            cur = None
    if cur is not None and t_end is not None:
        out.append((cur, t_end))
    return out


def overlap(iv, a, b):
    if a is None or b is None:
        return None
    return float(sum(max(0.0, min(y, b) - max(x, a)) for x, y in iv))


def expected_env(row):
    m_p, v = WORK[row['work']]
    return {'GRIP_PAYLOAD_KG': m_p, 'GRIP_SPEED': v,
            'GRIP_DETACH_DELAY_SEC': 4.0 if row['scen'] == 'd4' else 0.0,
            'MHE_THRUST_FAULT_GAIN': float(row['tgain']),
            'GRIP_UNCOMMANDED_LOSS': 'true' if row['fault'] == 'loss' else 'false',
            'GRIP_DETACH_JAM': 'true' if row['fault'] == 'jam' else 'false'}


# ---------------------------------------------------------------- 逐架
def parse_flight(row):
    fdir, st = row['flight_dir'], row['stamp']
    out = {k: row[k] for k in ('idx', 'block', 'cell_id', 'work', 'scen', 'fault', 'tgain',
                               'kind', 'replaces', 'stamp')}
    out['idx'] = int(out['idx'])
    out['m_p'], out['v'] = WORK[row['work']]
    if st in ('', 'NA'):
        out.update(eligible=False, ineligible_reason='startup-fail(no nmpc log)')
        return out
    N = wall_lines(os.path.join(fdir, f'grip_nmpc_{st}.log'))
    P = wall_lines(os.path.join(fdir, f'grip_proximity_{st}.log'))
    Mh = wall_lines(os.path.join(fdir, f'grip_mhe_{st}.log'))
    sp = os.path.join(fdir, f'grip_state_{st}.log')
    state_txt = open(sp, errors='replace').read() if os.path.exists(sp) else ''
    EN, EP = evts(N), evts(P)
    ev = lambda E, name: [w for n, w, _ in E if n == name]

    # ---- 资格:日志完整 ----
    rd = os.path.join(fdir, 'resid')
    streams = {k: one(rd, f'resid_{k}_*.csv') for k in ('imu', 'motor', 'internal', 'odom')}
    missing = [k for k, p in streams.items() if p is None or sum(1 for _ in open(p)) < 50]
    if not N or sum(1 for _, l in N if '[model-used]' in l) < 20:
        missing.append('nmpc')
    if not Mh:
        missing.append('mhe')
    if not P:
        missing.append('proximity')
    out['logs_missing'] = '+'.join(missing)

    # ---- 资格:配置与排程一致(溯源记录 + 节点启动行)----
    envf = read_env(os.path.join(fdir, f'provenance_{st}', 'experiment.env'))
    bad = []
    for k, v in expected_env(row).items():
        got = envf.get(k)
        if got is None:
            bad.append(f'{k}=<none>')
        elif isinstance(v, float):
            try:
                if abs(float(got) - v) > 1e-9:
                    bad.append(f'{k}={got}≠{v}')
            except ValueError:
                bad.append(f'{k}={got}≠{v}')
        elif got.lower() != v:
            bad.append(f'{k}={got}≠{v}')
    ntxt = ''.join(l for _, l in N[:4000])
    mtxt = ''.join(l for _, l in Mh[:2000])
    tg = float(row['tgain'])
    if N:
        if '[acc-det] release_confirm_source=acc' not in ntxt:
            bad.append('nmpc:no-acc-det')
        m = re.search(r'\[thrust-fault\] estimator-side thrust map gain=([\d.]+)', ntxt)
        got = float(m.group(1)) if m else 1.0
        if abs(got - tg) > 1e-6:
            bad.append(f'nmpc:tgain={got}')
    if Mh:
        if '[mhe-variant] reduced' not in mtxt:
            bad.append('mhe:not-reduced')
        m = re.search(r'thrust map gain=([\d.]+), motor', mtxt)
        got = float(m.group(1)) if m else 1.0
        if abs(got - tg) > 1e-6:
            bad.append(f'mhe:tgain={got}')
    out['config_bad'] = ';'.join(bad)

    # ---- 事件时刻(墙钟)----
    t_att = first(P, r'-> ATTACH')
    out['attached'] = bool(t_att is not None and 'ATTACHED' in state_txt)
    t_det = (ev(EP, 'sep') or [None])[0]
    if t_det is None:
        t_det = first(P, r'-> DETACH')
    t_lift = first(N, r'\| LIFT:')
    t_dyn = first(N, r'\| DYNAMIC:')
    t_cmd = (ev(EN, 'cmd') or [None])[0]
    t_inj = (ev(EN, 'inject_loss') or [None])[0]
    t_trig = t_cmd if t_cmd is not None else t_inj
    t_arm = (ev(EN, 'acc_loss_armed') or [None])[0]
    t_lost = first(N, r'LOST: PAYLOAD LOST')
    t_unres = (ev(EN, 'unresolved') or [None])[0]
    brake = [kv for n, _, kv in EN if n == 'brake']
    accepts = [(w, kv.get('pending') == '1') for n, w, kv in EN if n == 'accept']
    sf = [w for w, l in N if 'solve failed' in l]
    t_div = sf[DIV_SF] if len(sf) > DIV_SF else None
    t_end = N[-1][0] if N else None
    out.update(t_attach=t_att, t_lift=t_lift, t_dyn=t_dyn, t_cmd=t_cmd, t_inject=t_inj,
               t_detach=t_det, t_armed=t_arm, t_lost=t_lost, t_unresolved=t_unres)
    out['diverged'] = t_div is not None
    out['solve_failed'] = len(sf)
    out['phase_div'] = ('' if t_div is None else
                        'pre-lift' if t_lift is None or t_div < t_lift else
                        'lift' if t_dyn is None or t_div < t_dyn else
                        'carry' if t_trig is None or t_div < t_trig else 'post')

    # ---- 释放生命周期 ----
    carry_end = t_trig if t_trig is not None else (t_det if t_det else t_end)
    carry_acc = [w for w, _ in accepts if t_att and carry_end and t_att < w < carry_end]
    carry_lost = bool(t_lost and t_att and carry_end and t_lost < carry_end)
    out['carry_false_confirm'] = len(carry_acc) + int(carry_lost)
    pre_event = out['carry_false_confirm'] > 0
    out['armed'] = t_arm is not None and (t_lift is None or t_arm >= t_lift)
    out['armed_before_trigger'] = bool(t_arm and t_trig and t_arm < t_trig)
    post_acc = [w for w, _ in accepts if t_cmd is not None and w >= t_cmd - 1e-3]
    t_acc = post_acc[0] if post_acc else None
    out['complete'] = t_acc is not None
    out['unresolved'] = bool(t_unres and t_cmd and t_unres > t_cmd)
    out['delay_cmd'] = (t_acc - t_cmd) if t_acc else None
    out['delay_det'] = (t_acc - t_det) if (t_acc and t_det) else None
    out['early_confirm'] = bool(t_acc and (t_det is None or t_acc < t_det - 0.05))
    if row['fault'] == 'loss':
        out['lost_detected'] = bool(t_lost and t_inj and t_lost >= t_inj)
        out['lost_delay_det'] = (t_lost - t_det) if (out['lost_detected'] and t_det) else None
        out['lost_early'] = bool(t_lost and t_det and t_lost < t_det - 0.05)
        out['missed'] = bool(t_det and not out['lost_detected'])
        out['planned_drop_after_lost'] = bool(t_lost and t_cmd and t_cmd > t_lost)
    else:
        out['lost_detected'] = t_lost is not None
        out['lost_delay_det'] = None
        out['lost_early'] = False
        out['missed'] = bool(t_det and not out['complete'])
        out['planned_drop_after_lost'] = False
    out['jam_wrong_confirm'] = bool(row['fault'] == 'jam' and (t_acc or t_lost))
    out['brake_armed'] = bool(brake and brake[0].get('armed') in ('1', 'True', 'true'))

    # ---- 逐帧:跟踪误差、模型 ----
    fr = []
    for w, l in N:
        if '[model-used]' in l:
            m = MU.search(l)
            if m:
                ma = AE.search(l)
                ae = ma.group(1) if ma else 'nan'
                fr.append((w, float(m.group(1)), float(m.group(2)), float(m.group(3)),
                           int(m.group(4)), float(ae) if ae != 'nan' else np.nan))
    fr = np.array(fr) if fr else np.zeros((0, 6))

    def seg(a, b, col):
        if a is None or b is None or not len(fr):
            return np.array([])
        s = (fr[:, 0] > a) & (fr[:, 0] < b)
        v = fr[s, col]
        return v[np.isfinite(v)]
    t_ref = t_det if t_det is not None else (t_lost if t_lost else t_trig)
    end_carry = carry_end if carry_end else t_div
    lo = seg(t_dyn + 5 if t_dyn else None, end_carry, 1)
    out['rmse_loaded'] = rms(lo)
    out['peak_loaded'] = float(lo.max()) if len(lo) else None
    post = seg(t_ref, t_ref + POST_WIN if t_ref else None, 1)
    out['rmse_post'] = rms(post)
    out['peak_post'] = float(post.max()) if len(post) else None
    out['att_rms_loaded'] = rms(seg(t_dyn + 5 if t_dyn else None, end_carry, 5))
    # 无指令脱落:模型惯量是否清零、内环缩放是否回到 1
    out['geom_cleared'] = out['gain_reset'] = None
    if row['fault'] == 'loss' and t_lost:
        g_after = seg(t_lost + 3.0, t_lost + 8.0, 3)
        d_after = seg(t_lost + 0.5, t_lost + 3.0, 4)
        out['geom_cleared'] = bool(len(g_after) and np.max(np.abs(g_after)) < 1e-3
                                   and len(d_after) and d_after.min() == 1)
        oms = [(w, float(m.group(1))) for w, l in N for m in [OM.search(l)] if m]
        before = [v for w, v in oms if t_lost - 10 < w < t_lost]
        after = [v for w, v in oms if t_lost < w < t_lost + 10]
        out['gain_reset'] = bool(before and after and max(before) > 1.05 and min(after) <= 1.05)
    # 卡死 / 任何 UNRESOLVED:是否错误清模型、制动后是否稳定悬停
    out['jam_model_cleared'] = None
    out['hover_ok'] = None
    if row['fault'] == 'jam':
        d_post = seg(t_cmd, t_end, 4) if t_cmd else np.array([])
        out['jam_model_cleared'] = bool(len(d_post) and d_post.max() == 1)
    fo = streams.get('odom')
    if t_unres and fo:
        O = np.genfromtxt(fo, delimiter=',', names=True)
        if O.size > 10:
            tw = O['t_header_ns'] * 1e-9
            spd = np.sqrt(O['vb_x'] ** 2 + O['vb_y'] ** 2 + O['vb_z'] ** 2)
            s = tw > t_unres + 5.0
            out['hover_ok'] = bool(s.any() and spd[s].max() < 0.5 and not out['diverged'])
    # 饱和(flight-diag 5 s 窗)
    fd_l = []
    for w, l in N:
        m = FD.search(l)
        if m and t_dyn and end_carry and t_dyn + 5 < w < end_carry:
            fd_l.append([float(x) for x in m.groups()])
    fd_l = np.array(fd_l)
    out['tilt_max_loaded'] = float(fd_l[:, 0].max()) if len(fd_l) else None
    out['satP_loaded'] = float(fd_l[:, 3].mean()) if len(fd_l) else None
    # 加速度计检测器:带载/投放后 m_acc 偏差
    ac = np.array([(w, float(m.group(1)), float(m.group(2))) for w, l in N
                   for m in [AC.search(l)] if m and m.group(1) != 'nan']) \
        if N else np.zeros((0, 3))
    if not len(ac):
        ac = np.zeros((0, 3))

    def aseg(a, b, col):
        if a is None or b is None or not len(ac):
            return np.array([])
        s = (ac[:, 0] > a) & (ac[:, 0] < b)
        return ac[s, col]
    mt = M_B + out['m_p']
    la = aseg(t_lift + 12 if t_lift else None, carry_end, 1)
    pa = aseg(t_ref + 5 if t_ref else None, t_ref + POST_WIN if t_ref else None, 1)
    out['macc_loaded_pct'] = 100 * (float(np.median(la)) - mt) / mt if len(la) else None
    out['macc_post_pct'] = 100 * (float(np.median(pa)) - M_B) / M_B if len(pa) else None

    # ---- 估计器:质量误差、健康、求解 ----
    ms = np.asarray([float(x) for _, l in Mh for m in [TIMING.search(l)] if m and m.group(2)
                     for x in m.group(2).split(',')])
    out['mhe_solve_p95_ms'] = float(np.percentile(ms, 95)) if len(ms) else None
    out['mhe_nonzero_status'] = sum(1 for _, l in Mh if FAIL.search(l))
    hiv = intervals(Mh, r'\[reduced-health\] HEALTHY->UNHEALTHY',
                    r'\[reduced-health\] UNHEALTHY->HEALTHY', t_end)
    out['health_reasons'] = '+'.join(sorted({m.group(1) for _, l in Mh for m in
                                             [re.search(r'HEALTHY->UNHEALTHY reason=(\S+)', l)] if m}))
    out['health_unhealthy_loaded_s'] = overlap(hiv, t_lift, carry_end) if hiv else 0.0
    out['health_unhealthy_post_s'] = overlap(hiv, t_det, t_end) if hiv else 0.0
    niv = intervals(N, r'\[payload-health\] unhealthy/stale', r'\[payload-health\] healthy/fresh', t_end)
    out['nmpc_hold_loaded_s'] = overlap(niv, t_lift, carry_end) if niv else 0.0
    out['mass_loaded_bias_pct'] = out['mass_loaded_mae'] = None
    out['mass_post_bias_pct'] = out['mass_post_mae'] = None
    out['attach_conv_s'] = out['detach_conv_s'] = None
    fi = streams.get('internal')
    if fi and fo:
        it, od = read_csv_cols(fi), read_csv_cols(fo)
        if len(it.get('t_recv_ns', [])) and len(od.get('t_header_ns', [])):
            off = float(np.median(od['t_header_ns'] - od['t_recv_ns'])) * 1e-9
            t = it['t_recv_ns'] * 1e-9 + off
            mest = it['m_est']
            out['mass_loaded_bias_pct'], out['mass_loaded_mae'] = seg_err(
                t, mest, t_lift + 5.0 if t_lift else None, carry_end, mt)
            out['mass_post_bias_pct'], out['mass_post_mae'] = seg_err(
                t, mest, t_det + 5.0 if t_det else None, t_end, M_B)
            out['attach_conv_s'] = stable_delay(t, mest, t_lift, mt)
            out['detach_conv_s'] = stable_delay(t, mest, t_det, M_B)

    # ---- 资格 ----
    reason = ''
    if not out['attached']:
        reason = 'attach-fail'
    elif missing:
        reason = f'logs-missing({out["logs_missing"]})'
    elif bad:
        reason = f'config-mismatch({out["config_bad"]})'
    elif t_trig is None and not pre_event:
        reason = (f'pre-trigger-divergence({out["phase_div"]})' if out['diverged']
                  else 'no-trigger(infra/timeout)')
    elif t_div is not None and t_trig is not None and t_div < t_trig and not pre_event:
        reason = f'pre-trigger-divergence({out["phase_div"]})'
    out['eligible'] = reason == ''
    out['ineligible_reason'] = reason
    return out


# ---------------------------------------------------------------- 汇总
def load(bdir):
    rows = list(csv.DictReader(open(os.path.join(bdir, 'manifest.csv'))))
    return [parse_flight(r) for r in rows]


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def fmt(x, f='{:.2f}'):
    return '—' if x is None else f.format(x)


def cnt(F, key):
    return sum(1 for f in F if f.get(key))


def cell_key(f):
    return (f['fault'], f['tgain'], f['scen'], f['work'])


def summarize(F, bdir, quiet):
    info = dict(l.rstrip('\n').split('=', 1) for l in open(os.path.join(bdir, 'batch_info.txt')) if '=' in l)
    E = [f for f in F if f.get('eligible')]
    lines = [f"# 批次 {info.get('experiment')} — {os.path.basename(bdir)}", '',
             f"- 系统:{info.get('system', '')}",
             f"- 代码指纹 {info.get('fingerprint')},git {info.get('git_head', '')[:10]}"
             f"(运行时未提交改动 {info.get('git_dirty_runtime', '?')} 处),种子 {info.get('seed')}",
             f"- 已飞 {len(F)} 架,合格 {len(E)} 架;不合格:"
             + (', '.join(f"#{f['idx']} {f['ineligible_reason']}" for f in F if not f.get('eligible')) or '无'),
             '']
    groups = {}
    for f in E:
        groups.setdefault(cell_key(f), []).append(f)
    head = ('| 故障 | 增益 | 脱离 | 工况 | n | 完成 | 延迟(脱离后) s | 提前确认 | 带载误确认 | '
            'LOST 检出 | 卡死误确认 | UNRESOLVED | 悬停稳定 | 发散 | 求解失败 | '
            'RMSE 带载/投放后 m | 质量 MAE 带载 kg | 不健康 s 带载 |')
    lines += [head, '|' + '---|' * (head.count('|') - 1)]
    for k in sorted(groups):
        G = groups[k]
        fa, tg, sc, w = k
        if fa == 'loss':
            det = f"{cnt(G, 'lost_detected')}/{len(G)} ({fmt(med([g['lost_delay_det'] for g in G]))} s)"
        else:
            det = '—'
        lines.append(
            f"| {fa} | {tg} | {sc} | {w} | {len(G)} | {cnt(G, 'complete')} | "
            f"{fmt(med([g['delay_det'] for g in G]))} | {cnt(G, 'early_confirm')} | "
            f"{sum(g['carry_false_confirm'] for g in G)} | {det} | "
            f"{cnt(G, 'jam_wrong_confirm') if fa == 'jam' else '—'} | {cnt(G, 'unresolved')} | "
            f"{sum(1 for g in G if g['hover_ok'])}/{sum(1 for g in G if g['hover_ok'] is not None)} | "
            f"{cnt(G, 'diverged')} | {sum(g['solve_failed'] for g in G)} | "
            f"{fmt(med([g['rmse_loaded'] for g in G]), '{:.3f}')}/{fmt(med([g['rmse_post'] for g in G]), '{:.3f}')} | "
            f"{fmt(med([g['mass_loaded_mae'] for g in G]), '{:.4f}')} | "
            f"{fmt(med([g['health_unhealthy_loaded_s'] for g in G]), '{:.1f}')} |")
    lines += ['', '列说明:延迟 = 确认相对物理脱离的中位数;LOST 检出括号内为检出相对脱离的中位延迟;'
              '悬停稳定 = 进入 UNRESOLVED 的架次中制动 5 s 后速度 < 0.5 m/s 且未发散的比例;'
              '数值列为中位数,计数列为架次数。', '']
    open(os.path.join(bdir, 'summary.md'), 'w').write('\n'.join(lines))
    json.dump({'info': info, 'n_flown': len(F), 'n_eligible': len(E),
               'flights': F}, open(os.path.join(bdir, 'summary.json'), 'w'), indent=1, default=str)
    keys = sorted({k for f in F for k in f})
    with open(os.path.join(bdir, 'per_flight.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for f in sorted(F, key=lambda x: x['idx']):
            wr.writerow(f)
    if not quiet:
        print('\n'.join(lines))


def emit_reruns(bdir, F):
    sched = os.path.join(bdir, 'schedule.csv')
    S = list(csv.DictReader(open(sched)))
    if any(r['kind'] == 'rerun' for r in S):
        print(0)
        return
    bad = [f for f in F if not f.get('eligible')
           and f['ineligible_reason'].startswith(INFRA) and f['kind'] != 'rerun']
    nxt = max(int(r['idx']) for r in S) + 1
    by_idx = {r['idx']: r for r in S}
    with open(sched, 'a', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n')
        for f in bad:
            r = by_idx[str(f['idx'])]
            wr.writerow([nxt, r['block'], r['cell_id'], r['work'], r['scen'], r['fault'],
                         r['tgain'], 'rerun', r['idx']])
            nxt += 1
    print(len(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bdir')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--emit-infra-reruns', action='store_true')
    a = ap.parse_args()
    F = load(a.bdir)
    if a.emit_infra_reruns:
        emit_reruns(a.bdir, F)
        return
    summarize(F, a.bdir, a.quiet)


if __name__ == '__main__':
    sys.exit(main())
