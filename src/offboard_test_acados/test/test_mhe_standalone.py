#!/usr/bin/env python3
"""速度+质量降阶 MHE 的单元/集成测试,不需要 ROS/SITL。

覆盖:
  1 状态维度=4            2 四元数归一化          3 q 与 −q 同一旋转
  4 空机悬停恢复 m_B      5 0.15/0.30 kg 恢复      6 attach/drop 后重新收敛
  7 −10% 推力不再因旧下界跑满迭代                  8 −10% 触发质量健康异常(+10% 上界)
  9 非零求解状态保持上一帧有效质量(核心 + 节点两层)
  10 unhealthy 质量不进 NMPC(MHE 发布 + NMPC 真实消费代码)

合成数据用与被测代码独立的 numpy 积分器生成:输入在每个 0.1 s 区间零阶保持,
y[k] 是 t_k 时刻量测,U[k] 是区间 (t_{k-1}, t_k] 的平均推力 —— 与 mhe_node 的
缓冲约定一致(stage j 用 y_win[j] 与 u_win[j])。
"""

import os

import numpy as np
import pytest
from scipy.linalg import block_diag

os.environ.setdefault('ACADOS_SOURCE_DIR', '/home/clear/acados')
os.environ['LD_LIBRARY_PATH'] = (
    '/home/clear/acados/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))

from offboard_test_acados.mhe import model as rm  # noqa: E402
from offboard_test_acados.mhe import node as mn  # noqa: E402
from offboard_test_acados.mhe import solver_builder as rb  # noqa: E402
from offboard_test_acados.mhe.estimator import MassHealthMonitor, ReducedMassMHE  # noqa: E402
from offboard_test_acados.mhe.params import p as P  # noqa: E402
from offboard_test_acados.mhe.weights import M0_THETA, EventWeightSchedule  # noqa: E402
from offboard_test_acados.payload_estimate import PayloadEstimate  # noqa: E402

N, DT, G, KD, M_B = P.N, P.dt, P.g, P.kd, P.m_B


# ----------------------------------------------------------------- 工具
def _quat_from_rpy(r, p_, y):
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p_ / 2), np.sin(p_ / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def _rot_ref(q):
    """测试自带的旋转矩阵(标准公式,输入先归一化),与被测实现独立。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def simulate(mass_fn, duration, gain=1.0, seed=0, v_noise=0.01,
             flip_sign=False, tilt_deg=8.0):
    """返回 (Y[K,13], U[K,4], m_true[K])。控制器按空机质量 + vz 反馈出推力
    (不知道载荷),姿态做 ±tilt 的慢正弦机动,偏航匀速转。gain 只作用在**量测**
    推力上(估计器侧推力映射误差),真实受力不变。"""
    rng = np.random.default_rng(seed)
    K = int(round(duration / DT)) + 1
    sub = 10
    h = DT / sub
    v = np.zeros(3)
    Y = np.zeros((K, 13))
    U = np.zeros((K, 4))
    m_true = np.zeros(K)
    for k in range(K):
        t = k * DT
        a = np.deg2rad(tilt_deg)
        q = _quat_from_rpy(a * np.sin(0.7 * t), a * np.sin(0.43 * t + 1.0),
                           0.3 * t)
        q_meas = -q if (flip_sign and rng.random() < 0.5) else q
        Y[k, 3:6] = v + v_noise * rng.standard_normal(3)
        Y[k, 6:10] = q_meas
        m_true[k] = mass_fn(t)
        if k == K - 1:
            break
        r33 = _rot_ref(q)[2, 2]
        T = M_B * (G - 1.5 * v[2]) / r33      # 控制器只知道空机质量
        e3w = _rot_ref(q)[:, 2]
        for _ in range(sub):                  # 区间内 q、T 零阶保持
            m = mass_fn(t)
            f = lambda vv: (e3w * T - KD * vv) / m - np.array([0, 0, G])
            k1 = f(v); k2 = f(v + h/2*k1); k3 = f(v + h/2*k2); k4 = f(v + h*k3)
            v = v + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        U[k + 1, 0] = gain * T                # 区间 (t_k, t_{k+1}] 的平均推力
    U[0, 0] = U[1, 0]
    return Y, U, m_true


def windows(Y, U):
    for k in range(N, len(Y)):
        yield k, Y[k - N:k + 1], U[k - N + 1:k + 1]


@pytest.fixture(scope='module')
def solver():
    return rb.ensure_solver()


def run_estimator(solver, Y, U, events=(), seed_m=None):
    """跑完整条轨迹,返回 (m_hat[K], status[K], health_ok[K], reasons)。
    events: 质量事件的帧号(第一个事件后量测帧),模拟节点残差检测器的 notify。"""
    sched = EventWeightSchedule(M0_THETA)
    for j in range(N):     # 新 fixture 复用同一求解器:先恢复名义权重
        solver.cost_set(j, 'W', sched.W0_nom if j == 0 else sched.W_nom)
    core = ReducedMassMHE(solver, P, scheduler=sched)
    health = MassHealthMonitor(P)
    K = len(Y)
    m_hat = np.full(K, np.nan)
    status = np.full(K, -1)
    ok = np.ones(K, dtype=bool)
    reasons = []
    ev = set(events)
    for k, yw, uw in windows(Y, U):
        if k in ev:
            sched.notify_event(k)
        seed = seed_m if seed_m is not None else float(uw[-1][0] / G)
        r = core.step(yw, uw, k, m_seed=seed)
        status[k] = r.status
        m_hat[k] = r.m
        if r.ok:
            ok[k], why, _ = health.update(r.m)
            reasons.append(why)
    return m_hat, status, ok, reasons


# ----------------------------------------------------------------- 1-3
def test_reduced_state_dimension_is_four(solver):
    model = rm.build_model()
    assert model.x.shape == (4, 1)
    assert model.u.shape == (3, 1)
    assert model.p.shape == (5, 1)
    assert solver.acados_ocp.dims.nx == 4
    assert solver.acados_ocp.dims.nu == 3
    assert solver.acados_ocp.dims.np == 5


def test_quaternion_is_normalised():
    q = _quat_from_rpy(0.3, -0.2, 1.1)
    f = rm.thrust_axis_function()
    for s in (0.2, 1.0, 3.7):
        e3 = np.array(f(s * q)).ravel()
        np.testing.assert_allclose(e3, _rot_ref(q)[:, 2], atol=1e-12)
        np.testing.assert_allclose(rm.rotmat_np(s * q), _rot_ref(q), atol=1e-12)
    par = rm.stage_param(np.r_[np.zeros(6), 2.5 * q, np.zeros(3)], [20.0, 0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(par[:4]), 1.0, atol=1e-15)
    assert par[4] == 20.0
    assert rm.stage_param(np.zeros(13), [20.0]) is None     # 退化四元数被拒


def test_q_and_minus_q_give_same_rotation():
    f = rm.thrust_axis_function()
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.standard_normal(4)
        np.testing.assert_allclose(rm.rotmat_np(q), rm.rotmat_np(-q), atol=1e-14)
        np.testing.assert_allclose(np.array(f(q)), np.array(f(-q)), atol=1e-14)


# ----------------------------------------------------------------- 4-6
def test_empty_hover_recovers_body_mass(solver):
    Y, U, mt = simulate(lambda t: M_B, 12.0, seed=1)
    m, st, ok, _ = run_estimator(solver, Y, U)
    tail = m[-60:]
    assert np.all(st[N:] == 0)
    assert abs(np.median(tail) - M_B) / M_B < 0.005
    assert np.all(ok[-60:])


@pytest.mark.parametrize('mp', [0.15, 0.30])
def test_payload_mass_recovered(solver, mp):
    Y, U, mt = simulate(lambda t: M_B + mp, 12.0, seed=2)
    m, st, ok, _ = run_estimator(solver, Y, U)
    assert np.all(st[N:] == 0)
    assert abs(np.median(m[-60:]) - (M_B + mp)) < 0.01
    assert np.all(ok[-60:])


def test_reconverges_after_attach_and_drop(solver):
    t_att, t_drop, mp = 8.0, 20.0, 0.30
    mass = lambda t: M_B + (mp if t_att <= t < t_drop else 0.0)
    Y, U, mt = simulate(mass, 32.0, seed=4, flip_sign=True)
    k_att, k_drop = int(t_att / DT) + 1, int(t_drop / DT) + 1
    for events in ((k_att, k_drop), ()):      # 有/无事件降权都必须收敛
        m, st, ok, _ = run_estimator(solver, Y, U, events=events)
        assert np.all(st[N:] == 0)
        k = np.arange(len(m))
        loaded = (k >= k_att + 30) & (k < k_drop)
        empty = k >= k_drop + 30
        assert np.max(np.abs(m[loaded] - (M_B + mp))) < 0.02, events
        assert np.max(np.abs(m[empty] - M_B)) < 0.02, events


# ----------------------------------------------------------------- 7-8
def test_minus10pct_thrust_no_max_iter_at_old_bound(solver):
    Y, U, _ = simulate(lambda t: M_B, 12.0, seed=5, gain=0.90)
    m, st, ok, reasons = run_estimator(solver, Y, U)
    # 新边界:估计自由落到 ≈0.9·m_B,无任何非零状态(旧回放 107 次 status=2)
    assert np.all(st[N:] == 0), np.unique(st[N:])
    assert abs(np.median(m[-60:]) - 0.9 * M_B) < 0.01
    assert P.m_min_num < 0.9 * M_B - 0.2
    # 对照:同一数据套回旧下界 0.95·m_B → 估计被钉在下界(旧失效模式)
    old = 0.95 * M_B
    try:
        for j in range(1, N):
            solver.constraints_set(j, 'lbx', np.array([old]))
        m_old, st_old, _, _ = run_estimator(solver, Y, U)
        assert np.median(m_old[-60:]) <= old + 2e-3
    finally:
        for j in range(1, N):
            solver.constraints_set(j, 'lbx', np.array([P.m_min_num]))


def test_thrust_map_mismatch_triggers_health_flags(solver):
    Y, U, _ = simulate(lambda t: M_B, 10.0, seed=6, gain=0.90)
    m, st, ok, reasons = run_estimator(solver, Y, U)
    assert not np.any(ok[-40:])
    assert 'mass_below_body' in reasons[-1]
    Y, U, _ = simulate(lambda t: M_B + 0.30, 10.0, seed=7, gain=1.10)
    m, st, ok, reasons = run_estimator(solver, Y, U)
    assert np.all(st[N:] == 0)
    assert not np.any(ok[-40:])
    assert 'mass_above_envelope' in reasons[-1]


def test_health_monitor_hysteresis_and_bounds():
    h = MassHealthMonitor(P)
    assert h.update(M_B)[0]
    assert h.update(P.health_low_kg - 1e-3) == (False, 'mass_below_body', True)
    for _ in range(P.health_recover_frames - 1):
        assert not h.update(M_B)[0]
    assert h.update(M_B)[0]
    assert 'at_numeric_bound' in h.classify(P.m_min_num)
    assert 'at_numeric_bound' in h.classify(P.m_max)
    assert h.classify(float('nan')) == 'nonfinite'


# ----------------------------------------------------------------- 9
class _FailingSolver:
    """status 序列可控的假求解器(只实现核心用到的接口)。"""

    def __init__(self, statuses, m_val=2.30):
        self.statuses = list(statuses)
        self.m_val = m_val
        self.sets = []

    def set(self, j, field, val):
        self.sets.append((j, field))

    def cost_set(self, *a):
        pass

    def solve(self):
        return self.statuses.pop(0)

    def get_stats(self, name):
        return 100 if name == 'sqp_iter' else 0

    def get(self, j, field):
        return np.array([0.0, 0.0, 0.0, self.m_val])


def _const_window(m=M_B):
    q = np.array([1.0, 0, 0, 0])
    Y = np.tile(np.r_[np.zeros(6), q, np.zeros(3)], (N + 1, 1))
    U = np.tile([m * G, 0, 0, 0], (N, 1))
    return Y, U


def test_nonzero_status_holds_last_valid_mass_core():
    Y, U = _const_window()
    fs = _FailingSolver([0] + [2] * 6, m_val=2.30)
    core = ReducedMassMHE(fs, P)
    r = core.step(Y, U, 0, m_seed=M_B)
    assert r.ok and r.m == 2.30
    fs.m_val = 9.99                            # 失败时 get() 的值绝不能被采用
    seeds = []
    for i in range(1, 7):
        r = core.step(Y, U, i, seed_fn=lambda why: (seeds.append(why) or (2.10, True)))
        assert not r.ok and r.status == 2 and r.sqp_iter == 100
        assert r.m == 2.30 and core.m_est == 2.30
        if i == 5:
            assert r.reinit and r.reinit_seed == 2.10 and core.fail_streak == 0
            assert core.x0_bar[3] == 2.10      # 种子只进先验/初猜
    assert seeds == ['re-anchor'] and core.n_fail == 6 and core.n_reinit == 1


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, s):
        self.lines.append(('I', s))

    def warn(self, s):
        self.lines.append(('W', s))

    warning = warn

    def error(self, s):
        self.lines.append(('E', s))


class _NodeStub:
    """绑定新 mhe.node.MHENode 的真实方法,只桩 ROS 外壳。"""
    _solve_window = mn.MHENode._solve_window
    _seed_mass = mn.MHENode._seed_mass
    _window_balance_mass = mn.MHENode._window_balance_mass
    _inputs_fresh = mn.MHENode._inputs_fresh
    _frame_health = mn.MHENode._frame_health
    _publish_frame = mn.MHENode._publish_frame
    _now = mn.MHENode._now
    _log_periodic = lambda self: setattr(self, 'counter', self.counter + 1)

    def __init__(self, solver):
        self.now = 100.0
        self.log = _Log()
        self.solver = solver
        self.scheduler = EventWeightSchedule(M0_THETA)
        for j in range(N):
            solver.cost_set(j, 'W', self.scheduler.W0_nom if j == 0 else self.scheduler.W_nom)
        self.core = ReducedMassMHE(solver, P, scheduler=self.scheduler)
        self.health = MassHealthMonitor(P)
        self._mass_ok, self._mass_reason = True, ''
        self.m_est = M_B
        self.thrust_phys = M_B * G
        self.tau_phys = None
        self.y_buf, self.u_buf = [], []
        self.frames = 0
        self._in_transition = False
        self._solve_ms = []
        self._last_solve_ok = False
        self._last_solve_success_sec = None
        self.mass_pub, self.tau_phys_pub, self.payload_estimate_pub = _Pub(), _Pub(), _Pub()
        self.counter = 0
        self.payload_input_fresh_sec = 0.30
        self.payload_solution_fresh_sec = 0.30
        self._last_odom_rx_sec = self._last_motor_rx_sec = self._last_u_rx_sec = None
        self.payload_output_tau = 0.25
        self._m_payload_out = 0.0
        self._conf = 1.0
        self.no_payload_score_mode = 'mass'
        self._payload_conf_args = dict(mass_full=0.015, mass_zero=0.060,
                                       moment_full=0.0015, moment_zero=0.0060,
                                       inertia_full=0.0020, inertia_zero=0.0100)

    def get_logger(self):
        return self.log

    def get_clock(self):
        owner = self

        class _Now:
            @property
            def nanoseconds(self):
                return int(owner.now * 1e9)

        class _Clock:
            @staticmethod
            def now():
                return _Now()
        return _Clock()

    def tick(self, y13, u_row):
        self.now += DT
        stamp = self.get_clock().now().nanoseconds * 1e-9
        self._last_odom_rx_sec = self._last_motor_rx_sec = self._last_u_rx_sec = stamp
        self.y_buf.append(np.asarray(y13, dtype=float))
        if len(self.y_buf) > N + 1:
            self.y_buf.pop(0)
        self.u_buf.append(np.asarray(u_row, dtype=float))
        if len(self.u_buf) > N:
            self.u_buf.pop(0)
        self.thrust_phys = float(u_row[0])
        self.frames += 1
        if len(self.y_buf) == N + 1:
            self._solve_window()
        self._publish_frame()
        return PayloadEstimate.from_array(self.payload_estimate_pub.msgs[-1].data)


def test_node_nonzero_status_holds_mass_and_marks_unhealthy():
    Y, U = _const_window()
    fs = _FailingSolver([0] * 3 + [2] * 3, m_val=2.20)
    node = _NodeStub(fs)
    for _ in range(N):
        node.tick(Y[0], U[0])
    for _ in range(3):
        est = node.tick(Y[0], U[0])
    assert node.m_est == 2.20 and est.healthy
    n_pub = len(node.mass_pub.msgs)
    fs.m_val = 4.0
    for _ in range(3):
        est = node.tick(Y[0], U[0])
        assert node.m_est == 2.20 and not est.healthy
    assert len(node.mass_pub.msgs) == n_pub            # 失败帧不发 mass_pub
    warns = [s for lv, s in node.log.lines if lv == 'W' and 'MHE solve failed' in s]
    assert len(warns) == 3
    assert all('status=2' in s and 'sqp_iter=100' in s and 'streak' in s for s in warns)


# ----------------------------------------------------------------- 10
def test_unhealthy_mass_never_reaches_nmpc(solver):
    """−10% 推力:MHE 发布帧在质量带外时 healthy=0、mass_pub 不发带外值;
    NMPC 的真实消费代码对 unhealthy 帧保持模型。"""
    Y, U, _ = simulate(lambda t: M_B, 10.0, seed=8, gain=0.90)
    node = _NodeStub(solver)
    frames = [node.tick(Y[k], U[k]) for k in range(len(Y))]
    assert any(not f.healthy for f in frames[-40:])
    assert all(not f.healthy for f in frames[-40:])
    published = [m.data for m in node.mass_pub.msgs]
    assert all(v >= P.health_low_kg for v in published)
    assert node.m_est < P.health_low_kg                # 估计器内部照常是带外值
    assert any('[reduced-health] HEALTHY->UNHEALTHY reason=mass_below_body' in s
               for _, s in node.log.lines)

    from test_attach_j_bootstrap_standalone import _Stub as _NmpcStub
    from offboard_test_acados import acados_nmpc_node as an
    nmpc = _NmpcStub()
    nmpc.now_sec = 50.0
    nmpc.m_est = 2.30                                  # 上一帧可信模型
    bad = PayloadEstimate(1.85, np.zeros(2), np.zeros(2), np.zeros(3),
                          np.array([P.Jxx, P.Jyy, P.Jzz]), 1.0,
                          healthy=False, solution_age_sec=0.05)
    nmpc._payload_target = bad
    nmpc._payload_estimate_received = True
    nmpc._payload_estimate_rx_sec = 50.0
    an.AcadosNMPCNode._update_continuous_payload_model(nmpc)
    assert nmpc.m_est == 2.30
    good = PayloadEstimate(2.10, np.zeros(2), np.zeros(2), np.zeros(3),
                           np.array([P.Jxx, P.Jyy, P.Jzz]), 1.0,
                           healthy=True, solution_age_sec=0.05)
    nmpc._payload_target = good
    an.AcadosNMPCNode._update_continuous_payload_model(nmpc)
    assert nmpc.m_est < 2.30                           # 健康帧才被消费




# ----------------------------------------------------------------- 结构
def test_event_weights_have_reduced_dimensions():
    w = EventWeightSchedule(M0_THETA)
    assert w.W_nom.shape == (6, 6) and w.W0_nom.shape == (10, 10)
    np.testing.assert_allclose(np.diag(w.W0_pre)[6:], np.diag(P.Q0))
    np.testing.assert_allclose(np.diag(w.W_pre), 1e-4 * np.r_[np.diag(P.R), np.diag(P.Q)])


def test_mhe_codegen_cache_is_separate_from_nmpc():
    from offboard_test_acados import acados_model
    assert rm.MODEL_NAME != acados_model.MODEL_NAME
    assert rb.json_path().parent == rb.generated_root()
    assert rb.generated_root().name == 'acados_generated_files'
