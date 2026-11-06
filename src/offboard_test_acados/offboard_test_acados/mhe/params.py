#!/usr/bin/env python3
"""降阶 MHE 的全部参数(单一来源)。

取值与 ros2_ws_HJH 在 2026-11-05 主线(MHE_MODEL_VARIANT=reduced,FROZEN_ENV)下实际生效的
值逐项一致:窗口 N=20/dt=0.1、速度测量权重 1/0.05²、速度过程噪声 1e4、到达代价速度 1e3、
质量 0.1;数值质量边界 [0.70·m_B, 5.0];健康带 [m_B−0.10, m_B+包线+0.10],恢复 10 帧。
"""
import os

import numpy as np

from ..common import AirframeParams

_base = AirframeParams()


def _env_float(name, default):
    raw = os.environ.get(name, '')
    return float(raw) if raw.strip() else float(default)


class MHEParams:
    # --- 物理常数(与 NMPC 共用 common.AirframeParams)---
    g = _base.g
    kd = _base.kd
    Jxx, Jyy, Jzz = _base.Jxx, _base.Jyy, _base.Jzz
    m_B = float(_base.m)          # 空机质量

    # --- 窗口 ---
    N = int(os.environ['MHE_N']) if os.environ.get('MHE_N') else 20
    dt = 0.1

    # --- 维度 ---
    nx, nw, np_, mass_idx = 4, 3, 5, 3

    # --- 权重(Bryson)---
    std_vel = 0.05
    R = np.diag([1.0 / std_vel ** 2] * 3)
    Q = np.diag([1.0e4] * 3)
    Q0 = np.diag([1.0e3] * 3 + [0.1])

    # --- 质量数值边界(只防 1/m 奇异与跑飞)---
    m_min_num = _env_float('MHE_REDUCED_M_MIN', 0.70 * m_B)
    m_max = _env_float('MHE_REDUCED_M_MAX', 5.0)

    # --- 质量健康带(不使用任何真值)---
    mp_envelope = _env_float('MHE_MP_ENVELOPE', 0.30)     # 机架载荷包线
    health_low_kg = _env_float('MHE_REDUCED_HEALTH_LOW', m_B - 0.10)
    health_high_kg = _env_float('MHE_REDUCED_HEALTH_HIGH', m_B + mp_envelope + 0.10)
    health_recover_frames = max(1, int(_env_float('MHE_REDUCED_HEALTH_RECOVER', 10)))
    bound_tol = _env_float('MHE_REDUCED_BOUND_TOL', 2e-3)
    reinit_after = 5

    # --- 推力种子 ---
    seed_thrust_min = 1.0        # [N] 低于此推力视为未在飞

    # --- 对外发布的惯量(由质量代数推出)---
    rz_prior = _env_float('MHE_RZ_PRIOR', -0.27)   # 名义吊臂 [m]

    def weight_blocks(self):
        return {'R': self.R, 'Q': self.Q, 'Q0': self.Q0,
                'vel_slice': slice(0, 3), 'mass_idx': self.mass_idx}

    def describe(self):
        return (f'N={self.N} dt={self.dt} R_v=1/{self.std_vel}^2 '
                f'Q_v={self.Q[0, 0]:.0e} Q0_v={self.Q0[0, 0]:.0e} '
                f'Q0_m={self.Q0[3, 3]:.3g} lbx_m=[{self.m_min_num:.4f},{self.m_max:.4f}] '
                f'health_band=[{self.health_low_kg:.4f},{self.health_high_kg:.4f}] '
                f'recover={self.health_recover_frames}f rz_prior={self.rz_prior:+.3f}')

    def validate(self):
        if not (0.0 < self.m_min_num < self.health_low_kg < self.health_high_kg
                < self.m_max):
            raise ValueError(
                'MHE bounds must satisfy 0 < m_min < health_low < health_high < m_max; '
                f'got {self.m_min_num:.4f}, {self.health_low_kg:.4f}, '
                f'{self.health_high_kg:.4f}, {self.m_max:.4f}')


p = MHEParams()
p.validate()
