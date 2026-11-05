#!/usr/bin/env python3
"""Pure-state checks for the attach-time NMPC inertia bootstrap.

主线语义:attach 时按载荷包线上界给 J 一个安全地板,地板保持到投放指令,之后在
release_sec 内平滑交给 MHE 的 dJ;m 只来自 MHE。"""

import numpy as np

from offboard_test_acados import acados_nmpc_node as an
from offboard_test_acados.acados_params import p
from offboard_test_acados.payload_estimate import PayloadEstimate


class _Stub:
    def __init__(self):
        self.now_sec = 0.0
        self.grip_payload_envelope = 0.3
        self.grip_arm_d = 0.47
        self.attach_j_bootstrap_release_sec = 0.5
        self._attach_j_bootstrap_active = False
        self._attach_j_bootstrap_releasing = False
        self._attach_j_bootstrap_weight = 0.0
        self._attach_j_bootstrap_dj = 0.0
        self.grip_drop_pending = False
        self.grip_dropped = False

        self.payload_model_tau = 0.20
        self.payload_mass_slew = 0.60
        self.payload_dj_slew = 0.080
        self.payload_estimate_fresh_sec = 0.35
        self.no_payload_confidence = 1.0
        self._payload_estimate_received = False
        self._payload_estimate_rx_sec = None
        self._payload_estimate_health_prev = None
        self._payload_target = PayloadEstimate(
            p.m, np.zeros(2), np.zeros(2), np.zeros(3),
            np.array([p.Jxx, p.Jyy, p.Jzz]), 1.0)
        self.m_est = p.m
        self.dJ_est = 0.0
        self.omega_scale_cap = 5.0
        self.omega_scale_hysteresis = 0.08
        self.omega_scale_tau = 0.5
        self.omega_scale_rise_rate = 4.0
        self.omega_scale_recover_rate = 1.5
        self.omega_scale = 1.0
        self._omega_scale_target = 1.0
        self._omega_scale_stamp = None

    def get_logger(self):
        class _Logger:
            info = warn = staticmethod(lambda _message: None)
        return _Logger()

    def get_clock(self):
        owner = self
        class _Now:
            @property
            def nanoseconds(self):
                return int(owner.now_sec * 1e9)
        class _Clock:
            @staticmethod
            def now():
                return _Now()
        return _Clock()


_Stub._dJ_from_mp = an.AcadosNMPCNode._dJ_from_mp
_start = an.AcadosNMPCNode._start_attach_j_bootstrap
_update = an.AcadosNMPCNode._update_continuous_payload_model
_Stub._payload_estimate_is_fresh = an.AcadosNMPCNode._payload_estimate_is_fresh
_update_omega = an.AcadosNMPCNode._update_omega_scale

def test_bootstrap_uses_03kg_reduced_mass_inertia_only():
    state = _Stub()
    _start(state)
    expected = (p.m * 0.3 / (p.m + 0.3)) * 0.47 ** 2
    assert abs(state._attach_j_bootstrap_dj - expected) < 1e-12
    assert 0.057 < expected < 0.059

    for _ in range(round(1.5 / p.dt)):
        _update(state)
    assert state.dJ_est > 0.95 * expected
    assert state.m_est == p.m


def test_first_moment_alone_cannot_release_inertia_floor():
    state = _Stub()
    _start(state)
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = 0.0
    state.no_payload_confidence = 0.0
    state._payload_target = PayloadEstimate(
        p.m, np.array([0.0, 0.004]), np.array([0.0, 0.002]),
        np.zeros(3), np.array([p.Jxx, p.Jyy, p.Jzz]), 0.0, True, 0.0)
    for _ in range(20):
        _update(state)
    assert state._attach_j_bootstrap_active
    assert not state._attach_j_bootstrap_releasing


def test_unhealthy_mhe_cannot_release_bootstrap_or_move_mass():
    state = _Stub()
    _start(state)
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = 0.0
    dj = 0.030
    state._payload_target = PayloadEstimate(
        p.m + 0.15, np.array([0.0, 0.004]), np.array([0.0, 0.002]),
        np.array([dj, dj, 0.0]),
        np.array([p.Jxx + dj, p.Jyy + dj, p.Jzz]), 0.0, False, 0.0)
    for _ in range(round(1.0 / p.dt)):
        _update(state)
    assert state.m_est == p.m
    assert state._attach_j_bootstrap_active
    assert not state._attach_j_bootstrap_releasing


def test_unhealthy_mhe_freezes_bootstrap_handoff():
    state = _Stub()
    _start(state)
    for _ in range(round(1.5 / p.dt)):
        _update(state)
    state._attach_j_bootstrap_releasing = True
    state._attach_j_bootstrap_weight = 0.5
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = 0.0
    state._payload_target = PayloadEstimate(
        p.m + 0.15, np.zeros(2), np.zeros(2),
        np.array([0.03, 0.03, 0.0]),
        np.array([p.Jxx + 0.03, p.Jyy + 0.03, p.Jzz]),
        0.0, False, 0.0)
    before = (state.dJ_est, state._attach_j_bootstrap_weight)
    for _ in range(20):
        _update(state)
    assert (state.dJ_est, state._attach_j_bootstrap_weight) == before


def test_omega_scale_recovers_rate_limited_all_the_way_to_one():
    state = _Stub()
    state.omega_scale = 4.0
    state._omega_scale_target = 4.0
    state.dJ_est = 0.0
    _update_omega(state)  # establish timestamp and snap empty target to 1
    values = [state.omega_scale]
    for _ in range(round(5.0 / p.dt)):
        state.now_sec += p.dt
        before = state.omega_scale
        _update_omega(state)
        values.append(state.omega_scale)
        assert before - state.omega_scale <= (
            state.omega_scale_recover_rate * p.dt + 1e-8)
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert abs(state._omega_scale_target - 1.0) < 1e-12
    assert state.omega_scale < 1.001


def _loaded_target(state, m_total, dj):
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = state.now_sec
    state._payload_target = PayloadEstimate(
        m_total, np.zeros(2), np.zeros(2), np.array([dj, dj, 0.0]),
        np.array([p.Jxx + dj, p.Jyy + dj, p.Jzz]), 0.0, True, 0.0)


def test_hold_until_drop_keeps_floor_through_mhe_collapse():
    state = _Stub()
    _start(state)
    floor = state._attach_j_bootstrap_dj
    state.no_payload_confidence = 0.0
    _loaded_target(state, p.m + 0.3, 0.030)        # 持续带载证据也不提前交接
    for _ in range(round(3.0 / p.dt)):
        _update(state)
    assert state._attach_j_bootstrap_active
    assert not state._attach_j_bootstrap_releasing
    assert state.dJ_est > 0.95 * floor

    _loaded_target(state, p.m, 0.0)                # 8 字中 m̂ 塌到空机
    state.no_payload_confidence = 0.95
    for _ in range(round(3.0 / p.dt)):
        _update(state)
    assert state.dJ_est > 0.95 * floor


def test_hold_until_drop_releases_on_drop_command_without_evidence():
    state = _Stub()
    _start(state)
    for _ in range(round(1.5 / p.dt)):
        _update(state)
    state.grip_drop_pending = True
    state.no_payload_confidence = 1.0
    _loaded_target(state, p.m, 0.0)                # 投放后 MHE 已无载荷证据
    for _ in range(round(3.0 / p.dt)):
        _update(state)
    assert not state._attach_j_bootstrap_active
    assert state.dJ_est < 1e-3
