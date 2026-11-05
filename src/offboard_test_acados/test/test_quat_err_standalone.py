#!/usr/bin/env python3
"""NMPC 姿态误差形式开关 NMPC_QUAT_ERR(2026-11-05)的离线验收。

'vec'(默认)= q_ref* ⊗ q 的向量部分;'euclid' = q − q_ref(与 MHE 测量残差同形式)。
覆盖:默认维数与权重不变;euclid 的维数/零点/小角度幅值与 vec 一致;半球对齐的
翻号与连续性;euclid 在未对齐的翻号参考下残差范数≈2(说明对齐是必需的)。
纯 CasADi/numpy,不需要 acados 求解器与 ROS。
"""
import os
import subprocess
import sys

import casadi as cs
import numpy as np

sys.path.insert(0, '/home/clear/ros2_ws_HJH/src/offboard_test_acados')

from offboard_test_acados.common import align_quat_hemisphere, tracking_error_sym


def _yawq(psi):
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


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


def _err_fn(mode):
    x = cs.MX.sym('x', 13)
    xr = cs.MX.sym('xr', 13)
    return cs.Function('e', [x, xr], [tracking_error_sym(x, xr, mode)])


def _state(q):
    x = np.zeros(13)
    x[6:10] = q
    return x


def test_dimensions():
    assert _err_fn('vec').size1_out(0) == 12
    assert _err_fn('euclid').size1_out(0) == 13


def test_unknown_mode_raises():
    try:
        tracking_error_sym(cs.MX.sym('x', 13), cs.MX.sym('xr', 13), 'geodesic')
    except ValueError:
        return
    raise AssertionError('unknown quat_err mode must raise')


def test_zero_at_reference_and_small_angle_scale():
    fv, fe = _err_fn('vec'), _err_fn('euclid')
    qr = _rotq([0.3, -0.5, 0.8], 0.7)
    assert np.allclose(np.array(fe(_state(qr), _state(qr))).ravel(), 0.0)
    rng = np.random.default_rng(3)
    for _ in range(20):
        ang = 0.02 * rng.random()
        q = _qmul(qr, _rotq(rng.standard_normal(3), ang))
        ev = np.array(fv(_state(q), _state(qr))).ravel()[6:9]
        ee = np.array(fe(_state(q), _state(qr))).ravel()[6:10]
        # both ≈ |dtheta|/2 for small angles, so the same L_att applies
        assert abs(np.linalg.norm(ev) - ang / 2) < 1e-4
        assert abs(np.linalg.norm(ee) - ang / 2) < 1e-4


def test_alignment_flips_and_keeps_continuity():
    q0 = _yawq(np.radians(170))
    seq = np.stack([_yawq(np.radians(a)) for a in np.linspace(170, 200, 21)], 1)
    seq[:, 7:] *= -1.0                      # emulate a sign flip mid-horizon
    al = align_quat_hemisphere(seq, q0)
    assert np.dot(al[:, 0], q0) > 0
    assert all(np.dot(al[:, k], al[:, k - 1]) > 0 for k in range(1, al.shape[1]))
    for k in range(al.shape[1]):            # same rotations, only signs changed
        assert np.allclose(np.abs(al[:, k]), np.abs(seq[:, k]))
    assert np.shares_memory(al, seq) is False


def test_euclid_needs_alignment():
    fe = _err_fn('euclid')
    q = _yawq(np.radians(179))
    flipped = -q
    raw = np.array(fe(_state(q), _state(flipped))).ravel()[6:10]
    assert abs(np.linalg.norm(raw) - 2.0) < 1e-9          # zero angle, norm 2
    al = align_quat_hemisphere(flipped.reshape(4, 1), q)[:, 0]
    assert np.allclose(np.array(fe(_state(q), _state(al))).ravel()[6:10], 0.0)


def _params_in_subprocess(env_value):
    env = dict(os.environ)
    if env_value is None:
        env.pop('NMPC_QUAT_ERR', None)
    else:
        env['NMPC_QUAT_ERR'] = env_value
    code = ('import sys; sys.path.insert(0, "/home/clear/ros2_ws_HJH/src/offboard_test_acados");'
            'from offboard_test_acados.acados_params import p, scaled_stage_terminal_W;'
            'W, We = scaled_stage_terminal_W(10.0, 0.283);'
            'print(p.quat_err, p.Q.shape[0], p.P.shape[0], W.shape[0], We.shape[0])')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                         text=True)
    return out.returncode, out.stdout.split(), out.stderr


def test_weight_dimensions_follow_mode():
    rc, o, _ = _params_in_subprocess(None)
    assert rc == 0 and o == ['vec', '12', '12', '16', '12']
    rc, o, _ = _params_in_subprocess('euclid')
    assert rc == 0 and o == ['euclid', '13', '13', '17', '13']
    rc, _, err = _params_in_subprocess('bogus')
    assert rc != 0 and 'NMPC_QUAT_ERR' in err


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
