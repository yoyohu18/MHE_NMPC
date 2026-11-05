#!/usr/bin/env python3
# 降阶 MHE 的窗口求解核心 + 质量健康监视器(纯 Python,无 ROS)。
#
# mhe_node 在 MHE_MODEL_VARIANT=reduced 时把窗口装载/求解/失败重锚委托给这里,
# 单测直接驱动同一份代码(而不是复制一个"测试版"),避免桩漂移。
#
# 失败语义(与 full 对齐,并补上 full 没有的两条):
#   * status≠0 或解含非有限值 → 不更新 m_est(保持上一帧有效质量),ok=False;
#   * 连续失败达到 reinit_after(=5,同 full 的 _fail_streak>=5)→ 到达先验与初猜
#     用当前窗口量测 + 推力种子重新初始化;**对外的 m_est 仍保持上一帧有效值**
#     (full 会把 m_est 直接改成种子,这里不这样做:种子不是一次有效求解);
#   * 每次结果都带 status 与 SQP 迭代数,供节点日志记录。

import math
import time
from dataclasses import dataclass

import numpy as np

from .model import stage_param
from .params import p as P0


@dataclass
class ReducedStepResult:
    status: int
    sqp_iter: int
    solve_ms: float
    ok: bool
    m: float               # ok 时为新解;否则为保持的上一帧有效质量
    fail_streak: int
    reinit: bool = False
    reinit_seed: float = float('nan')
    nonfinite: bool = False


class MassHealthMonitor:
    """不使用真值的质量健康检查。

    带外判据(任一命中即 unhealthy,**立即**生效):
      mass_below_body   m̂ < health_low  —— 推力映射可能低估(或地面支撑等暂态)
      mass_above_envelope m̂ > health_high —— 超出机架载荷规格(推力映射可能高估)
      at_numeric_bound  m̂ 贴 lbx/ubx    —— 数值边界在起作用,解不可信
      nonfinite
    恢复:连续 recover_frames 帧回到带内才重新 healthy(迟滞,防抖)。
    """

    def __init__(self, params=None):
        P = params or P0
        self.low = P.health_low_kg
        self.high = P.health_high_kg
        self.m_min = P.m_min_num
        self.m_max = P.m_max
        self.tol = P.bound_tol
        self.recover_frames = P.health_recover_frames
        self.reset()

    def reset(self):
        self.healthy = True
        self.reason = ''
        self._good = 0
        self.unhealthy_frames = 0
        self.transitions = 0

    def classify(self, m):
        if m is None or not math.isfinite(m):
            return 'nonfinite'
        why = []
        if m < self.low:
            why.append('mass_below_body')
        if m > self.high:
            why.append('mass_above_envelope')
        if m <= self.m_min + self.tol or m >= self.m_max - self.tol:
            why.append('at_numeric_bound')
        return '+'.join(why)

    def update(self, m):
        """返回 (healthy, reason, changed)。"""
        prev = self.healthy
        bad = self.classify(m)
        if bad:
            self.healthy = False
            self.reason = bad
            self._good = 0
        elif not self.healthy:
            self._good += 1
            if self._good >= self.recover_frames:
                self.healthy = True
                self.reason = ''
                self._good = 0
        if not self.healthy:
            self.unhealthy_frames += 1
        changed = self.healthy != prev
        if changed:
            self.transitions += 1
        return self.healthy, self.reason, changed


class ReducedMassMHE:
    """速度+质量降阶 MHE 的滑动窗口求解器状态。

    窗口约定与 full 完全相同:y_win 是最近 N+1 帧 13 维量测(世界系速度在 [3:6]、
    四元数在 [6:10]),u_win 是最近 N 行已知输入 [T_phys, ...];stage j 的代价用
    y_win[j],参数用 (q_j, T_j)。stage N 不进代价。
    """

    def __init__(self, solver, params=None, scheduler=None):
        self.solver = solver
        self.P = params or P0
        self.scheduler = scheduler
        self.reinit_after = self.P.reinit_after
        self.reset()

    def reset(self):
        self.x0_bar = None
        self.x_guess = None
        self.m_est = None
        self.fail_streak = 0
        self.n_solves = 0
        self.n_fail = 0
        self.n_reinit = 0
        self.last_status = 0

    def reset_window(self):
        """输入流中断时节点清窗:先验/初猜作废,保留统计与上一帧有效质量。"""
        self.x0_bar = None
        self.x_guess = None
        self.fail_streak = 0

    def _clip(self, m):
        return float(np.clip(float(m), self.P.m_min_num, self.P.m_max))

    def _state(self, y13, m):
        return np.array([y13[3], y13[4], y13[5], m], dtype=float)

    def initialize(self, y_win, m_seed):
        N = self.P.N
        m_seed = self._clip(m_seed)
        self.x0_bar = self._state(y_win[0], m_seed)
        self.x_guess = [self._state(y_win[min(j, N)], m_seed) for j in range(N + 1)]
        if self.m_est is None:
            self.m_est = m_seed

    def load_window(self, y_win, u_win):
        """把当前窗口写进求解器(yref / p / 初猜)。不含权重调度。"""
        N = self.P.N
        if len(y_win) < N + 1 or len(u_win) < N:
            raise ValueError(f'window too short: y={len(y_win)} u={len(u_win)}')
        if self.x0_bar is None:
            raise RuntimeError('ReducedMassMHE.initialize() must be called first')
        z3 = np.zeros(self.P.nw)
        for j in range(N):
            y = np.asarray(y_win[j], dtype=float)
            par = stage_param(y, u_win[j])
            if par is None:
                raise ValueError(f'invalid quaternion/thrust at stage {j}')
            if j == 0:
                self.solver.set(0, 'yref', np.concatenate([y[3:6], z3, self.x0_bar]))
            else:
                self.solver.set(j, 'yref', np.concatenate([y[3:6], z3]))
            self.solver.set(j, 'p', par)
            self.solver.set(j, 'x', self.x_guess[j])
            self.solver.set(j, 'u', z3)
        self.solver.set(N, 'x', self.x_guess[N])

    def solve_loaded(self, y_win, seed_fn=None):
        """求解已装载窗口。seed_fn(why)->(m_seed, from_thrust) 仅在重新初始化时用。"""
        N = self.P.N
        t0 = time.perf_counter()
        status = int(self.solver.solve())
        ms = (time.perf_counter() - t0) * 1e3
        try:
            it = int(self.solver.get_stats('sqp_iter'))
        except Exception:      # noqa: BLE001 —— 统计量缺失不影响估计
            it = -1
        self.n_solves += 1
        self.last_status = status
        held = self.m_est if self.m_est is not None else float('nan')
        if status != 0:
            return self._fail(status, it, ms, y_win, seed_fn, held)
        xs = [np.asarray(self.solver.get(j, 'x'), dtype=float) for j in range(N + 1)]
        if not all(np.all(np.isfinite(x)) for x in xs):
            r = self._fail(status, it, ms, y_win, seed_fn, held)
            r.nonfinite = True
            return r
        self.m_est = float(xs[N][self.P.mass_idx])
        self.x0_bar = xs[1].copy()
        self.x_guess = [xs[min(j + 1, N)].copy() for j in range(N + 1)]
        self.fail_streak = 0
        return ReducedStepResult(status, it, ms, True, self.m_est, 0)

    def _fail(self, status, it, ms, y_win, seed_fn, held):
        self.n_fail += 1
        self.fail_streak += 1
        res = ReducedStepResult(status, it, ms, False, held, self.fail_streak)
        if self.fail_streak >= self.reinit_after:
            m_seed, from_thrust = (seed_fn('re-anchor') if seed_fn is not None
                                   else (held, False))
            if not from_thrust or m_seed is None or not math.isfinite(m_seed):
                m_seed = held if math.isfinite(held) else self.P.m_B
            m_seed = self._clip(m_seed)
            N = self.P.N
            self.x0_bar = self._state(y_win[0], m_seed)
            self.x_guess = [self._state(y_win[min(j, N)], m_seed)
                            for j in range(N + 1)]
            self.fail_streak = 0
            self.n_reinit += 1
            res.reinit = True
            res.reinit_seed = m_seed
        return res

    def step(self, y_win, u_win, fc, seed_fn=None, m_seed=None):
        """测试/离线用的一步:必要时初始化 → 装载 → 事件降权 → 求解。"""
        if self.x0_bar is None:
            if m_seed is None:
                m_seed = seed_fn('initial window')[0] if seed_fn else self.P.m_B
            self.initialize(y_win, m_seed)
        self.load_window(y_win, u_win)
        if self.scheduler is not None:
            self.scheduler.apply(self.solver, fc)
        return self.solve_loaded(y_win, seed_fn)
