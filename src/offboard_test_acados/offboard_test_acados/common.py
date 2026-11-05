"""Shared airframe constants and math helpers for the acados package."""

import math

import casadi as cs
import numpy as np


class AirframeParams:
    """Bare-airframe and actuator constants used by both NMPC and MHE."""

    m = 2.0643
    g = 9.81
    Jxx = 0.0142
    Jyy = 0.0142
    Jzz = 0.0210
    kd = 0.05
    Tmin = 0.5
    Tmax = 2 * m * g
    tau_max = 0.5
    tau_psi = 0.2


QUAT_ERR_MODES = ('vec', 'euclid')


def align_quat_hemisphere(q_seq, q_anchor):
    """Flip the sign of each column of ``q_seq`` (4 x K) so that it lies in the
    same hemisphere as its predecessor, starting from ``q_anchor``.

    q and -q are the same rotation, but the 'euclid' attitude error q - q_ref is
    not sign-invariant: a reference that flips sign (e.g. at a yaw wrap of +-pi)
    would turn a zero-angle error into a residual of norm 2. Chaining the
    alignment along the horizon keeps the reference continuous. Returns a copy.
    """
    q_seq = np.array(q_seq, dtype=float, copy=True)
    prev = np.asarray(q_anchor, dtype=float)
    for k in range(q_seq.shape[1]):
        if float(np.dot(q_seq[:, k], prev)) < 0.0:
            q_seq[:, k] = -q_seq[:, k]
        prev = q_seq[:, k]
    return q_seq


def tracking_error_sym(x, xr, quat_err='vec'):
    """Return the symbolic position, velocity, attitude, and rate error.

    quat_err='vec' (default): attitude error = vector part of q_ref* (x) q (3-dim,
    sign-invariant after squaring). quat_err='euclid': q - q_ref (4-dim), the same
    form as the MHE measurement residual; the caller must keep q_ref in the
    hemisphere of q (see align_quat_hemisphere). For small angles both have
    magnitude |dtheta|/2, so the same tolerance L_att applies.
    """
    if quat_err not in QUAT_ERR_MODES:
        raise ValueError(f'quat_err={quat_err!r} not in {QUAT_ERR_MODES}')
    ep = x[0:3] - xr[0:3]
    ev = x[3:6] - xr[3:6]
    q = x[6:10]
    qr = xr[6:10]
    if quat_err == 'euclid':
        return cs.vertcat(ep, ev, q - qr, x[10:13] - xr[10:13])
    qrc = cs.vertcat(qr[0], -qr[1], -qr[2], -qr[3])
    pw, px, py, pz = qrc[0], qrc[1], qrc[2], qrc[3]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    prod_q = cs.vertcat(
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )
    return cs.vertcat(ep, ev, prod_q[1:4], x[10:13] - xr[10:13])


def quat_to_rotmat(qw, qx, qy, qz):
    """Convert a scalar-first quaternion to a body-to-world rotation matrix."""
    return np.array([
        [qw**2 + qx**2 - qy**2 - qz**2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), qw**2 - qx**2 + qy**2 - qz**2, 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw**2 - qx**2 - qy**2 + qz**2],
    ])


def quat_to_euler(qw, qx, qy, qz):
    """Convert a scalar-first quaternion to roll, pitch, and yaw."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert roll, pitch, and yaw to a normalized scalar-first quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ])
    return q / np.linalg.norm(q)


def build_reference(t, r=1.0, w=0.3, z_hover=3.0, dz=0.0,
                    hover_time=2.0, ramp_time=4.0):
    """Build the smooth-ramp circular reference used by the NMPC node."""
    if t < hover_time:
        return np.concatenate([
            np.array([r, 0.0, z_hover + dz]),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
        ])

    tc = t - hover_time
    if tc < ramp_time:
        s = tc / ramp_time
        alpha = 10 * s**3 - 15 * s**4 + 6 * s**5
        alpha_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / ramp_time
    else:
        alpha = 1.0
        alpha_dot = 0.0

    a = w * tc
    ca, sa = math.cos(a), math.sin(a)
    x_f, y_f, z_f = r * ca, r * sa, z_hover + dz * ca
    vx_f, vy_f, vz_f = -r * w * sa, r * w * ca, -dz * w * sa
    ax_f, ay_f = -r * w * w * ca, -r * w * w * sa
    x0, y0, z0 = r, 0.0, z_hover + dz
    x = x0 + alpha * (x_f - x0)
    y = y0 + alpha * (y_f - y0)
    z = z0 + alpha * (z_f - z0)
    vx = alpha_dot * (x_f - x0) + alpha * vx_f
    vy = alpha_dot * (y_f - y0) + alpha * vy_f
    vz = alpha_dot * (z_f - z0) + alpha * vz_f
    ax = alpha_dot * vx_f + alpha * ax_f
    ay = alpha_dot * vy_f + alpha * ay_f

    denom = vx * vx + vy * vy
    if denom > 1e-4:
        yaw = math.atan2(vy, vx)
        yaw_rate = (vx * ay - vy * ax) / denom
    else:
        yaw = 0.0
        yaw_rate = 0.0
    return np.concatenate([
        np.array([x, y, z]),
        np.array([vx, vy, vz]),
        euler_to_quat(0.0, 0.0, yaw),
        np.array([0.0, 0.0, yaw_rate]),
    ])


def build_reference_window(t_start, n, dt, r=1.0, w=0.3,
                           z_hover=3.0, dz=0.0):
    """Build a horizon of circular reference states."""
    xref = np.zeros((13, n + 1))
    for i in range(n + 1):
        xref[:, i] = build_reference(t_start + i * dt, r, w, z_hover, dz)
    return xref
