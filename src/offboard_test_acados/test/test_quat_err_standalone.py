#!/usr/bin/env python3
"""NMPC 姿态误差(q_ref* ⊗ q 的向量部分)的离线验收:维数、零点、±q 不变性、
小角度幅值 |δθ|/2(与 Bryson 容许量 L_att 的口径一致)。"""
import casadi as cs
import numpy as np

from offboard_test_acados.acados_params import p
from offboard_test_acados.common import tracking_error_sym


def _rotq(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _err_fn():
    x = cs.MX.sym('x', 13)
    xr = cs.MX.sym('xr', 13)
    return cs.Function('e', [x, xr], [tracking_error_sym(x, xr)])


def _state(q):
    x = np.zeros(13)
    x[6:10] = q
    return x


def test_dimensions_match_weights():
    assert _err_fn().size1_out(0) == 12
    assert p.Q.shape == (12, 12) and p.P.shape == (12, 12) and p.nq_err == 3


def test_zero_at_reference_and_sign_invariant_cost():
    f = _err_fn()
    qr = _rotq([0.3, -0.5, 0.8], 0.7)
    assert np.allclose(np.array(f(_state(qr), _state(qr))).ravel(), 0.0)
    q = _qmul(qr, _rotq([1.0, 0.2, -0.4], 0.1))
    e_pos = np.array(f(_state(q), _state(qr))).ravel()[6:9]
    e_neg = np.array(f(_state(-q), _state(qr))).ravel()[6:9]
    # q 与 −q 是同一个姿态:误差只差符号,平方代价相同
    assert np.allclose(e_pos, -e_neg)


def test_small_angle_scale():
    f = _err_fn()
    qr = _rotq([0.3, -0.5, 0.8], 0.7)
    rng = np.random.default_rng(3)
    for _ in range(20):
        ang = 0.02 * rng.random()
        q = _qmul(qr, _rotq(rng.standard_normal(3), ang))
        e = np.array(f(_state(q), _state(qr))).ravel()[6:9]
        assert abs(np.linalg.norm(e) - ang / 2) < 1e-4
