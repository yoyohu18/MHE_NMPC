#!/usr/bin/env python3
"""Offline analyses answering the second thesis review (2026-11-06).

Input: the 96-flight F/R paired batch (mhe_variant_ab_20261105_181925/main).
Everything here is open-loop replay on recorded data; nothing touches the runtime.

A  reference-point ablation: reduced MHE replayed on the body-origin velocity
   (as flown) and on the composite-CoM velocity v_c = v_o + R (w x r_c).
B  terminal-measurement ablation: reduced MHE with a velocity residual on y_N.
C  simple mass-estimator baselines on the same samples.
"""
import csv, math, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/clear/ros2_ws_HJH/src/scripts/gripper')
import replay_reduced_mhe as RR  # noqa: E402

BATCH = Path('/home/clear/ros2_ws_HJH/nmpc_test_results/mhe_variant_ab_20261105_181925/main')
OUT = Path(__import__('os').environ.get('REVIEW_OUT', '/tmp/review_replay'))
OUT.mkdir(parents=True, exist_ok=True)
M_B, G, N, DT, KD = RR.M_B, RR.G, RR.N, RR.DT, RR.KD
RZ = -0.27
RR.M_MIN = 0.70 * M_B            # deployed R numerical box


def build(term: bool, tag: str):
    import casadi as cs
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    from scipy.linalg import block_diag
    x = cs.MX.sym('x', 4); w = cs.MX.sym('w', 3); par = cs.MX.sym('p', 5)
    v, m = x[0:3], x[3]
    qw, qx, qy, qz, T = par[0], par[1], par[2], par[3], par[4]
    r3 = cs.vertcat(2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), qw*qw-qx*qx-qy*qy+qz*qz)
    md = AcadosModel(); md.name = f'rev_{tag}'
    md.x, md.u, md.p = x, w, par
    md.f_expl_expr = cs.vertcat((r3*T - KD*v)/m - cs.vertcat(0, 0, G) + w, 0.0)
    md.cost_y_expr_0 = cs.vertcat(v, w, x); md.cost_y_expr = cs.vertcat(v, w)
    o = AcadosOcp(); o.model = md
    cg = OUT / 'codegen' / tag
    o.code_gen_opts.code_export_directory = str(cg / 'c_generated_code')
    o.solver_options.N_horizon = N; o.solver_options.tf = N*DT
    o.cost.cost_type_0 = o.cost.cost_type = 'NONLINEAR_LS'
    o.cost.W_0 = block_diag(RR.R_V, RR.Q_V, RR.Q0_VM); o.cost.W = block_diag(RR.R_V, RR.Q_V)
    o.cost.yref_0 = np.zeros(10); o.cost.yref = np.zeros(6)
    o.cost.cost_type_e = 'LINEAR_LS'
    if term:
        o.cost.Vx_e = np.hstack([np.eye(3), np.zeros((3, 1))]); o.cost.W_e = RR.R_V
        o.cost.yref_e = np.zeros(3)
    else:
        o.cost.Vx_e = np.zeros((0, 4)); o.cost.yref_e = np.zeros(0)
    o.parameter_values = np.array([1.0, 0, 0, 0, M_B*G])
    o.constraints.idxbx = np.array([3]); o.constraints.lbx = np.array([RR.M_MIN])
    o.constraints.ubx = np.array([RR.M_MAX])
    so = o.solver_options
    so.integrator_type = 'ERK'; so.sim_method_num_stages = 4; so.sim_method_num_steps = 1
    so.qp_solver = 'PARTIAL_CONDENSING_HPIPM'; so.qp_solver_cond_N = N
    so.hessian_approx = 'GAUSS_NEWTON'; so.nlp_solver_type = 'SQP'
    so.globalization = 'MERIT_BACKTRACKING'
    cg.mkdir(parents=True, exist_ok=True)
    return AcadosOcpSolver(o, json_file=str(cg / 'ocp.json'))


def replay(s, solver, term: bool):
    """Same loop as RR.replay, plus the optional y_N reference."""
    solver.reset(); sch = RR.WeightSchedule(); sch.reset(solver)
    n = len(s.t_ns); out = np.full(n, np.nan); st = np.zeros(n, int)
    m_est = M_B; x0b = xg = None; fails = 0; armf = armb = None
    for k in range(n):
        out[k] = m_est
        if k and s.attached_flag[k] != s.attached_flag[k-1]:
            armf, armb = k, float(s.thrust[k])
        if armf is not None:
            c = abs(float(s.thrust[k]) - armb) > 1.5; to = k - armf > 30
            if c or to:
                sch.notify(k if c else armf); armf = armb = None
        if k < N:
            continue
        vw = s.vel[k-N:k+1]; qw = s.quat[k-N:k+1]; tw = s.thrust[k-N+1:k+1]
        if x0b is None:
            ms = RR.window_seed(vw, qw, tw)
            x0b = np.r_[vw[0], ms]; xg = [np.r_[vw[min(j, N)], ms] for j in range(N+1)]; m_est = ms
        solver.set(0, 'yref', np.r_[vw[0], np.zeros(3), x0b])
        for j in range(N):
            if j:
                solver.set(j, 'yref', np.r_[vw[j], np.zeros(3)])
            solver.set(j, 'x', xg[j]); solver.set(j, 'u', np.zeros(3))
            solver.set(j, 'p', np.r_[qw[j], tw[j]])
        solver.set(N, 'x', xg[N])
        if term:
            solver.set(N, 'yref', vw[N])
        sch.apply(solver, k)
        r = int(solver.solve()); st[k] = r
        if r != 0:
            fails += 1
            if fails >= 5:
                x0b = np.r_[vw[0], m_est]; xg = [np.r_[vw[min(j, N)], m_est] for j in range(N+1)]; fails = 0
            continue
        xs = [np.asarray(solver.get(j, 'x')) for j in range(N+1)]
        m_est = float(xs[N][3]); x0b = xs[1].copy(); xg = [xs[min(j+1, N)].copy() for j in range(N+1)]
        fails = 0
    return out, st


def fnum(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(row):
    fdir = Path(row['flight_dir']); rd = fdir / 'resid'
    s = RR.sample_flight(rd)
    od = RR.read_csv(RR.one_file(rd, 'resid_odom_*.csv'))
    off = float(np.median(od['t_header_ns'] - od['t_recv_ns'])) * 1e-9
    t = s.t_ns * 1e-9 + off
    ot = od['t_recv_ns'].astype(np.int64)
    oi = np.searchsorted(ot, s.t_ns, side='right') - 1
    wb = np.column_stack([od[k][oi] for k in ('wb_x', 'wb_y', 'wb_z')])
    ta, td = fnum(row['t_attach']), fnum(row['t_detach'])
    mp = float(row['m_p'])
    att = np.zeros(len(t), int)
    if ta is not None:
        att = ((t >= ta) & ((t < td) if td is not None else True)).astype(int)
    s.attached_flag = att
    mtrue = M_B + mp * att
    # composite CoM below the odometry point (airframe CoM = base_link = IMU)
    rc = np.zeros((len(t), 3)); rc[:, 2] = (mp * att / (M_B + mp * att)) * RZ
    vc = s.vel.copy()
    for k in range(len(t)):
        vc[k] = s.vel[k] + RR.quat_rot(s.quat[k]) @ np.cross(wb[k], rc[k])
    return s, t, wb, rc, vc, mtrue


def windows(row, t):
    g = lambda k: fnum(row[k])
    tl, tdy, tc, td = g('t_lift'), g('t_dyn'), g('t_cmd'), g('t_detach')
    end_l = min(x for x in (tc, td) if x is not None)
    W = {'loaded': (t >= tl + 5) & (t < end_l),
         'fig8': (t >= tdy + 5) & (t < end_l),
         'post': (t >= td + 5) if td is not None else np.zeros(len(t), bool)}
    return W, tl, td


def metrics(m, mtrue, W, t, tl, td, mp):
    r = {}
    for k, w in W.items():
        e = (m - mtrue)[w & np.isfinite(m)]
        r[f'mae_{k}'] = float(np.mean(np.abs(e))) if len(e) else math.nan
        r[f'bias_{k}_pct'] = float(100*np.median(e)/np.median(mtrue[w])) if len(e) else math.nan
    r['conv_lift'] = RR.stable_delay(t, m, tl, M_B + mp)
    r['conv_det'] = RR.stable_delay(t, m, td, M_B) if td is not None else math.nan
    return r


# ---------------- simple baselines ----------------
def bl_window(s):            # sliding 2 s vertical force balance (the window seed)
    n = len(s.t_ns); m = np.full(n, np.nan)
    for k in range(N, n):
        m[k] = RR.window_seed(s.vel[k-N:k+1], s.quat[k-N:k+1], s.thrust[k-N+1:k+1])
    return m


def bl_lpf(s, tau=1.0):      # low-pass T cos / low-pass (g + a_z)
    n = len(s.t_ns); a = math.exp(-DT/tau)
    num = den = None; m = np.full(n, np.nan)
    for k in range(1, n):
        c = RR.quat_rot(s.quat[k])[2, 2]
        y1 = s.thrust[k] * c; y2 = G + (s.vel[k, 2] - s.vel[k-1, 2]) / DT
        num = y1 if num is None else a*num + (1-a)*y1
        den = y2 if den is None else a*den + (1-a)*y2
        m[k] = np.clip(num / max(den, 0.5*G), RR.M_MIN, RR.M_MAX)
    return m


def bl_rls(s, lam=0.95):     # RLS on theta = 1/m, 3 rows per tick
    n = len(s.t_ns); th = 1/M_B; P = 1e-2; m = np.full(n, np.nan)
    for k in range(1, n):
        r3 = RR.quat_rot(s.quat[k-1])[:, 2]
        phi = r3*s.thrust[k] - KD*s.vel[k-1]
        y = (s.vel[k] - s.vel[k-1])/DT + np.array([0, 0, G])
        P /= lam
        for i in range(3):
            K = P*phi[i] / (1 + phi[i]*P*phi[i]); th += K*(y[i] - phi[i]*th); P -= K*phi[i]*P
        m[k] = np.clip(1/th, RR.M_MIN, RR.M_MAX)
    return m


def bl_ekf(s, sig_m=0.05, sig_a=0.3, sig_v=0.05):   # x=[v, m], random-walk mass
    n = len(s.t_ns); x = np.r_[s.vel[0], M_B]; P = np.diag([0.01]*3 + [0.1])
    Q = np.diag([(sig_a*DT)**2]*3 + [sig_m**2*DT]); Rm = np.eye(3)*sig_v**2
    H = np.hstack([np.eye(3), np.zeros((3, 1))]); m = np.full(n, np.nan)
    for k in range(1, n):
        r3 = RR.quat_rot(s.quat[k-1])[:, 2]; T = s.thrust[k]
        f = r3*T - KD*x[:3]
        x = np.r_[x[:3] + DT*(f/x[3] - np.array([0, 0, G])), x[3]]
        F = np.eye(4); F[:3, :3] -= DT*KD/x[3]*np.eye(3); F[:3, 3] = -DT*f/x[3]**2
        P = F@P@F.T + Q
        S = H@P@H.T + Rm; K = P@H.T@np.linalg.inv(S)
        x = x + K@(s.vel[k] - x[:3]); P = (np.eye(4) - K@H)@P
        x[3] = np.clip(x[3], RR.M_MIN, RR.M_MAX); m[k] = x[3]
    return m


def transport_stats(t, wb, rc, s, W):
    """magnitude of the neglected terms in the loaded figure-eight."""
    w = W['fig8']
    if w.sum() < 10:
        return {}
    al = np.gradient(wb, t, axis=0)
    acc = np.array([np.cross(al[k], -rc[k]) + np.cross(wb[k], np.cross(wb[k], -rc[k])) for k in range(len(t))])
    dv = np.linalg.norm(s.vel - (s.vel + np.array([RR.quat_rot(s.quat[k]) @ np.cross(wb[k], rc[k]) for k in range(len(t))])), axis=1)
    return {'dv_rms': float(np.sqrt(np.mean(dv[w]**2))), 'dv_max': float(dv[w].max()),
            'acc_rms': float(np.sqrt(np.mean(np.sum(acc[w]**2, 1)))),
            'acc_z_mean': float(np.mean(acc[w, 2])),
            'w_rms': float(np.sqrt(np.mean(np.sum(wb[w]**2, 1)))),
            'alpha_rms': float(np.sqrt(np.mean(np.sum(al[w]**2, 1))))}


def main():
    rows = list(csv.DictReader(open(BATCH / 'per_flight.csv')))
    man = {r['idx']: r for r in csv.DictReader(open(BATCH / 'manifest.csv'))}
    sel = [dict(r, **{'flight_dir': man[r['idx']]['flight_dir'], 'mhe': man[r['idx']]['mhe']})
           for r in rows if r['kind'] == 'main']
    only = sys.argv[1] if len(sys.argv) > 1 else 'R'
    sel = [r for r in sel if r['mhe'] == only or only == 'all']
    sv = {'base': (build(False, 'base'), False), 'term': (build(True, 'term'), True)}
    res = []
    for i, row in enumerate(sel):
        t0 = time.time()
        s, t, wb, rc, vc, mtrue = load(row)
        W, tl, td = windows(row, t); mp = float(row['m_p'])
        rec = {'idx': int(row['idx']), 'mhe': row['mhe'], 'pair': row.get('pair_id', ''),
               'm_p': mp, 'v': float(row['v']), 'scen': row['scen']}
        online = s.online_m.copy()
        est = {'online': online}
        for tag, (so, term) in sv.items():
            est[f'R_{tag}'], _ = replay(s, so, term)
        vo = s.vel; s.vel = vc
        est['R_com'], _ = replay(s, sv['base'][0], False)
        est['R_com_term'], _ = replay(s, sv['term'][0], True)
        s.vel = vo
        est['WFB'] = bl_window(s); est['LPF'] = bl_lpf(s); est['RLS'] = bl_rls(s); est['EKF'] = bl_ekf(s)
        for k, m in est.items():
            for mk, mv in metrics(m, mtrue, W, t, tl, td, mp).items():
                rec[f'{k}.{mk}'] = mv
        rec.update({f'tr.{k}': v for k, v in transport_stats(t, wb, rc, s, W).items()})
        rec['replay_vs_online_mae'] = float(np.nanmean(np.abs(est['R_base'] - online)[W['loaded']]))
        res.append(rec)
        print(f"[{i+1}/{len(sel)}] idx={rec['idx']} {row['mhe']} v={rec['v']} mp={mp} "
              f"fig8 MAE online={rec['online.mae_fig8']:.4f} base={rec['R_base.mae_fig8']:.4f} "
              f"com={rec['R_com.mae_fig8']:.4f} term={rec['R_term.mae_fig8']:.4f} "
              f"WFB={rec['WFB.mae_fig8']:.4f} EKF={rec['EKF.mae_fig8']:.4f} ({time.time()-t0:.1f}s)", flush=True)
    json.dump(res, open(OUT / f'rev_results_{only}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
