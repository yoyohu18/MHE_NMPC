#!/usr/bin/env python3
"""速度+质量降阶 MHE 的 acados 模型。

  x = [v_x, v_y, v_z, m]            世界系速度 + 总质量(机体+载荷)
  u = w (3)                          速度过程噪声(acados 'u' 槽)
  p = [q_w, q_x, q_y, q_z, T_phys]   已知输入:实测姿态 + 电机反算总推力

  v_dot = (R(q)e3·T_phys / |q|² − k_d·v)/m − g·e3 + w,   m_dot = 0

R(q) 的每个元素是 q 的二次型 → 除以 |q|² 即在符号层归一化,且 R(q)=R(−q) 严格成立;
代价里没有任何四元数残差(姿态是已知输入,不是状态)。
"""
import math

import casadi as cs
import numpy as np
from acados_template import AcadosModel

from .params import p as P

MODEL_NAME = 'offboard_test_acados_mhe_reduced_vm'


# ------------------------------------------------------------- 姿态工具
def normalize_quat(q):
    """返回单位四元数(numpy)。模长过小视为无效,返回 None。"""
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not (n > 1e-6) or not np.all(np.isfinite(q)):
        return None
    return q / n


def rotmat_np(q):
    """R(q/|q|),与模型同一公式(二次型 / |q|²)。q 与 −q 结果相同。"""
    q = np.asarray(q, dtype=float).reshape(4)
    qq = float(q @ q)
    if not qq > 1e-12:
        raise ValueError('quaternion norm too small')
    qw, qx, qy, qz = q
    return np.array([
        [qw*qw + qx*qx - qy*qy - qz*qz, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), qw*qw - qx*qx + qy*qy - qz*qz, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz],
    ]) / qq


def _thrust_axis_sym(q):
    """R(q)e3 / |q|²(符号)。"""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    qq = cs.fmax(qw*qw + qx*qx + qy*qy + qz*qz, 1e-12)
    return cs.vertcat(2*(qx*qz + qw*qy),
                      2*(qy*qz - qw*qx),
                      qw*qw - qx*qx - qy*qy + qz*qz) / qq


def thrust_axis_function():
    """casadi Function q(4) -> R(q)e3,供单测核对符号模型。"""
    q = cs.SX.sym('q', 4)
    return cs.Function('thrust_axis', [q], [_thrust_axis_sym(q)])


def build_model() -> AcadosModel:
    x = cs.MX.sym('x', P.nx)          # [v(3), m]
    w = cs.MX.sym('w', P.nw)
    par = cs.MX.sym('p', P.np_)        # [q(4), T_phys]
    v, m = x[0:3], x[3]
    thrust = par[4]
    vdot = (_thrust_axis_sym(par[0:4]) * thrust - P.kd * v) / m \
        - cs.vertcat(0.0, 0.0, P.g) + w

    model = AcadosModel()
    model.name = MODEL_NAME
    model.x = x
    model.u = w
    model.p = par
    model.f_expl_expr = cs.vertcat(vdot, 0.0)
    # stage 0: [速度测量(3); 过程噪声(3); 到达代价(4)];stage 1..N-1: [测量; 噪声]
    model.cost_y_expr_0 = cs.vertcat(v, w, x)
    model.cost_y_expr = cs.vertcat(v, w)
    return model


def stage_param(y13, u_row):
    """由 13 维量测帧(世界系速度、四元数在 [6:10])与已知输入行 [T,...] 组 p。
    四元数无效时返回 None。"""
    q = normalize_quat(np.asarray(y13, dtype=float)[6:10])
    if q is None:
        return None
    T = float(np.asarray(u_row, dtype=float)[0])
    if not math.isfinite(T):
        return None
    return np.concatenate([q, [T]])
