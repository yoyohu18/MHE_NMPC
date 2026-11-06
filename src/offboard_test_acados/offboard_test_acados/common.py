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
    tau_max = 0.5
    tau_psi = 0.2


def tracking_error_sym(x, xr):
    """Return the symbolic position, velocity, attitude, and rate error.

    The attitude error is the vector part of q_ref* (x) q (3-dim, sign-invariant
    after squaring); for small angles its magnitude is |dtheta|/2.
    """
    ep = x[0:3] - xr[0:3]
    ev = x[3:6] - xr[3:6]
    q = x[6:10]
    qr = xr[6:10]
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
