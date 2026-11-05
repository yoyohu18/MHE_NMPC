#!/usr/bin/env python3
"""Pure-state checks for the attach-time NMPC inertia bootstrap."""

import numpy as np

from offboard_test_acados import acados_nmpc_node as an
from offboard_test_acados.acados_params import p
from offboard_test_acados.payload_estimate import PayloadEstimate


class _Stub:
    def __init__(self):
        self.now_sec = 0.0
        self.control_mode = 'mhe'
        self.grip_payload_envelope = 0.3
        self.grip_arm_d = 0.47
        self.attach_j_bootstrap_enable = True
        self.attach_j_bootstrap_confirm_frames = max(1, round(0.5 / p.dt))
        self.attach_j_bootstrap_release_sec = 0.5
        self.attach_j_bootstrap_min_dj = 0.010
        self._attach_j_bootstrap_active = False
        self._attach_j_bootstrap_releasing = False
        self._attach_j_bootstrap_weight = 0.0
        self._attach_j_bootstrap_dj = 0.0
        self._attach_j_bootstrap_confirm_frames = 0

        self.payload_model_tau = 0.20
        self.payload_mass_slew = 0.60
        self.payload_moment_slew = 0.080
        self.payload_dj_slew = 0.080
        self.no_payload_conf_threshold = 0.90
        self.payload_estimate_fresh_sec = 0.35
        self.no_payload_confidence = 1.0
        self.no_payload_conf_hold_frames = 5
        self._no_payload_conf_frames = 0
        self._no_payload_latched = True
        self._payload_estimate_received = False
        self._payload_estimate_rx_sec = None
        self._payload_estimate_health_prev = None
        self._payload_target = PayloadEstimate(
            p.m, np.zeros(2), np.zeros(2), np.zeros(3),
            np.array([p.Jxx, p.Jyy, p.Jzz]), 1.0)
        self.m_est = p.m
        self.s_est = np.zeros(2)
        self.c_est = np.zeros(2)
        self.dJ_est = 0.0
        self.omega_scale_enable = True
        self.omega_scale_source = 'djest'
        self.omega_scale_cap = 5.0
        self.omega_scale_hysteresis = 0.08
        self.omega_scale_tau = 0.5
        self.omega_scale_rise_rate = 4.0
        self.omega_scale_recover_rate = 1.5
        self.omega_scale = 1.0
        self._omega_scale_target = 1.0
        self._omega_scale_stamp = None
        self.gripper_mode = False
        self.drop_pending_confirmation = False
        # omega_scale 的 dJ 下界(2026-09-22 为 L1 臂补的内环补偿):主线默认
        # 0.0 = 逐位 no-op,本文件测的就是主线路径。
        self._omega_scale_dj_floor = 0.0
        # c_xy 冻结默认关(2026-09-17 翻回),_apply_cxy_freeze 首行即直通。
        self.cxy_freeze_in_maneuver = False
        # _accept_empty 里唯一没有 getattr 兜底的位,产品默认 False(见 624 行)。
        self.payload_unresolved = False

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
_confirm_empty = an.AcadosNMPCNode._confirm_no_payload_if_persistent
_update_omega = an.AcadosNMPCNode._update_omega_scale
# 非被测对象,但都在被测路径上被调用,绑真方法即可:
# _apply_cxy_freeze 在 cxy_freeze_in_maneuver=False 时首行直通;
# _check_drop_unresolved 首行 getattr('grip_drop_pending', False) 后即 return。
_Stub._apply_cxy_freeze = an.AcadosNMPCNode._apply_cxy_freeze
_Stub._check_drop_unresolved = an.AcadosNMPCNode._check_drop_unresolved
# 清除逻辑门控:内部 getattr(release_baseline,'P') 兜底 → 主线恒 True。
_Stub._confidence_accept_allowed = an.AcadosNMPCNode._confidence_accept_allowed
# accept 路径终点:本文件 test_unhealthy_frame_breaks_empty_confidence_hold
# 的第二段就是要走到这里 latch 空载,必须是真方法。
_Stub._accept_empty = an.AcadosNMPCNode._accept_empty
# 2026-11-01 确认链证据来源:release_confirm_source 缺省 'mhe' → 前者首行 return、
# 后者等价 _payload_estimate_is_fresh。
_Stub._update_acc_confirm = an.AcadosNMPCNode._update_acc_confirm
_Stub._release_evidence_fresh = an.AcadosNMPCNode._release_evidence_fresh


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
    assert np.array_equal(state.s_est, np.zeros(2))
    assert np.array_equal(state.c_est, np.zeros(2))


def test_persistent_mhe_dj_blends_out_bootstrap_without_a_step():
    state = _Stub()
    _start(state)
    for _ in range(round(1.5 / p.dt)):
        _update(state)

    # Also covers a payload lighter than the 0.3kg safety envelope: once MHE
    # supplies coherent non-zero dJ evidence, the conservative floor fades out.
    mhe_dj = 0.030
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = 0.0
    state.no_payload_confidence = 0.0
    state._payload_target = PayloadEstimate(
        p.m + 0.15, np.array([0.0, 0.004]),
        np.array([0.0, 0.004]) / (p.m + 0.15),
        np.array([mhe_dj, mhe_dj, 0.0]),
        np.array([p.Jxx + mhe_dj, p.Jyy + mhe_dj, p.Jzz]), 0.0,
        True, 0.0)

    samples = []
    for _ in range(round(3.0 / p.dt)):
        before = state.dJ_est
        _update(state)
        samples.append(state.dJ_est)
        assert abs(state.dJ_est - before) <= state.payload_dj_slew * p.dt + 1e-12

    assert not state._attach_j_bootstrap_active
    assert abs(state.dJ_est - mhe_dj) < 2e-4
    assert state.m_est > p.m       # MHE mass is still consumed normally
    assert np.linalg.norm(state.s_est) > 0.0


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
    assert np.array_equal(state.s_est, np.zeros(2))
    assert state._attach_j_bootstrap_active
    assert not state._attach_j_bootstrap_releasing


def test_unhealthy_frame_breaks_empty_confidence_hold():
    state = _Stub()
    state._no_payload_latched = False
    state._no_payload_conf_frames = 4
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = 0.0
    state.no_payload_confidence = 1.0
    state._payload_target = PayloadEstimate(
        p.m, np.zeros(2), np.zeros(2), np.zeros(3),
        np.array([p.Jxx, p.Jyy, p.Jzz]), 1.0, False, 0.0)
    _confirm_empty(state)
    assert state._no_payload_conf_frames == 0
    assert not state._no_payload_latched

    state._payload_target = PayloadEstimate(
        p.m, np.zeros(2), np.zeros(2), np.zeros(3),
        np.array([p.Jxx, p.Jyy, p.Jzz]), 1.0, True, 0.0)
    for _ in range(state.no_payload_conf_hold_frames):
        _confirm_empty(state)
    assert state._no_payload_latched


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


if __name__ == '__main__':
    test_bootstrap_uses_03kg_reduced_mass_inertia_only()
    test_persistent_mhe_dj_blends_out_bootstrap_without_a_step()
    test_first_moment_alone_cannot_release_inertia_floor()
    test_unhealthy_mhe_cannot_release_bootstrap_or_move_mass()
    test_unhealthy_frame_breaks_empty_confidence_hold()
    test_unhealthy_mhe_freezes_bootstrap_handoff()
    test_omega_scale_recovers_rate_limited_all_the_way_to_one()
    print('all passed')


def _loaded_target(state, m_total, dj):
    state._payload_estimate_received = True
    state._payload_estimate_rx_sec = state.now_sec
    state._payload_target = PayloadEstimate(
        m_total, np.zeros(2), np.zeros(2), np.array([dj, dj, 0.0]),
        np.array([p.Jxx + dj, p.Jyy + dj, p.Jzz]), 0.0, True, 0.0)


def test_hold_until_drop_keeps_floor_through_mhe_collapse():
    state = _Stub()
    state.attach_j_bootstrap_hold_until_drop = True
    _start(state)
    floor = state._attach_j_bootstrap_dj
    state.no_payload_confidence = 0.0
    _loaded_target(state, p.m + 0.3, 0.030)        # 持续带载证据:默认档会交接
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
    state.attach_j_bootstrap_hold_until_drop = True
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
