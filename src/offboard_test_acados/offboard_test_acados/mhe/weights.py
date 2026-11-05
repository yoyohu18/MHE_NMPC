#!/usr/bin/env python3
"""事件降权时间表(降阶 MHE 维度)。

残差检测器确认一次质量突变后,把窗口内"事件前"的 stage 的测量/过程噪声权重乘
10**theta[0](默认 1e-4),到达代价块保持名义;事件滑出窗口后恢复名义权重。
theta 四个分量的语义与原 M0 规则相同:
  theta[0] log10 事件前 stage 的 R/Q 缩放          (默认 -4)
  theta[1] log10 事件后 stage 的速度测量缩放        (默认 0)
  theta[2] 事件滑出窗口后再延长的帧数               (默认 0)
  theta[3] log10 过渡期到达代价质量锚缩放           (默认 0)
"""

import numpy as np
from scipy.linalg import block_diag

from .params import p as P

M0_THETA = np.array([-4.0, 0.0, 0.0, 0.0])


class EventWeightSchedule:
    def __init__(self, theta=M0_THETA, params=P):
        self.N = params.N
        self.theta = np.asarray(theta, dtype=float)
        d_pre = 10.0 ** self.theta[0]
        r_vel = 10.0 ** self.theta[1]
        self.extra = int(round(self.theta[2]))
        q0m = 10.0 ** self.theta[3]
        R, Q, Q0 = params.R, params.Q, params.Q0
        R_post = R * r_vel                       # R 只有速度块
        Q0_t = Q0.copy()
        Q0_t[params.mass_idx, params.mass_idx] *= q0m
        self.W_nom = block_diag(R, Q)
        self.W0_nom = block_diag(R, Q, Q0)
        self.W_pre = block_diag(d_pre * R, d_pre * Q)
        self.W_post = block_diag(R_post, Q)
        self.W0_pre = block_diag(d_pre * R, d_pre * Q, Q0_t)
        self.W0_post = block_diag(R_post, Q, Q0_t)
        self.event_frame = None
        self._dirty = False

    def notify_event(self, next_frame):
        """记录第一个事件后测量帧的全局序号。"""
        self.event_frame = int(next_frame)

    def apply(self, solver, fc):
        """每次求解前调用,fc = 窗口末端帧序号。返回是否处于过渡期。"""
        N = self.N
        if self.event_frame is None:
            return False
        n_pre = N - (fc - self.event_frame)
        if n_pre >= N:
            return True        # 代价里还没有事件后测量:保持名义
        if n_pre <= -self.extra:
            if self._dirty:
                solver.cost_set(0, 'W', self.W0_nom)
                for j in range(1, N):
                    solver.cost_set(j, 'W', self.W_nom)
                self._dirty = False
            self.event_frame = None
            return False
        solver.cost_set(0, 'W', self.W0_pre if n_pre > 0 else self.W0_post)
        for j in range(1, N):
            solver.cost_set(j, 'W', self.W_pre if j < n_pre else self.W_post)
        self._dirty = True
        return True

    def reset_nominal(self, solver):
        solver.cost_set(0, 'W', self.W0_nom)
        for j in range(1, self.N):
            solver.cost_set(j, 'W', self.W_nom)
        self.event_frame = None
        self._dirty = False
