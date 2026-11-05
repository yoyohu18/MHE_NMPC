#!/usr/bin/env python3
"""NMPC 释放确认路径的测试桩(绑定 acados_nmpc_node 的真实方法,只桩 ROS 外壳)。
原在 ros2_ws_HJH 的 test_drop_confirm_standalone.py 中,抽出供多个测试共用。"""
import numpy as np

from offboard_test_acados import acados_nmpc_node as mn_nmpc
from offboard_test_acados.mhe.params import p as mhe_p  # noqa: F401
from offboard_test_acados.payload_estimate import PayloadEstimate  # noqa: F401


class _Clock:
    def __init__(self, owner):
        self.owner = owner

    def now(self):
        class _Now:
            pass
        v = _Now()
        v.nanoseconds = int(self.owner.now_sec * 1e9)
        return v


class _Logger:
    def __init__(self):
        self.msgs = []

    def info(self, m):
        self.msgs.append(m)

    warn = error = info


class _NMPCStub:
    _confirm_no_payload_if_persistent = (
        mn_nmpc.AcadosNMPCNode._confirm_no_payload_if_persistent)
    _check_drop_unresolved = mn_nmpc.AcadosNMPCNode._check_drop_unresolved
    _begin_unresolved_hover = mn_nmpc.AcadosNMPCNode._begin_unresolved_hover
    _grip_unresolved_ref = mn_nmpc.AcadosNMPCNode._grip_unresolved_ref
    _grip_unresolved_ref_window = (
        mn_nmpc.AcadosNMPCNode._grip_unresolved_ref_window)
    _payload_estimate_is_fresh = (
        mn_nmpc.AcadosNMPCNode._payload_estimate_is_fresh)
    _confidence_accept_allowed = (
        mn_nmpc.AcadosNMPCNode._confidence_accept_allowed)
    _accept_empty = mn_nmpc.AcadosNMPCNode._accept_empty
    _update_acc_confirm = mn_nmpc.AcadosNMPCNode._update_acc_confirm
    _release_evidence_fresh = mn_nmpc.AcadosNMPCNode._release_evidence_fresh
    _update_acc_loss_watchdog = (
        mn_nmpc.AcadosNMPCNode._update_acc_loss_watchdog)
    _reset_acc_loss_watchdog = (
        mn_nmpc.AcadosNMPCNode._reset_acc_loss_watchdog)
    _handle_uncommanded_payload_loss = (
        mn_nmpc.AcadosNMPCNode._handle_uncommanded_payload_loss)

    def __init__(self):
        self._payload_estimate_received = True
        self._payload_estimate_rx_sec = 100.0
        self._payload_target = PayloadEstimate(
            m_total=2.06, s_xy=np.zeros(2), c_xy=np.zeros(2),
            dJ_diag=np.zeros(3), J_diag=np.ones(3),
            no_payload_confidence=1.0, healthy=True, solution_age_sec=0.0)
        self._payload_estimate_health_prev = True
        self.payload_estimate_fresh_sec = 0.30
        self.no_payload_confidence = 1.0
        self.no_payload_conf_threshold = 0.90
        self.no_payload_conf_hold_frames = 20
        self._no_payload_conf_frames = 0
        self._no_payload_latched = False
        self.control_mode = 'mhe'
        self.gripper_mode = True
        self.grip_drop_pending = False
        self.grip_drop_done = False
        # 2026-09-15:acados_nmpc_node 新增故障注入开关(默认关),_grip_drop_phase 会读它
        self.grip_uncommanded_loss = False
        self.grip_dropped = False
        self.now_sec = 100.0
        # unresolved 路径(2026-09-05)
        self.drop_unresolved_timeout_sec = 12.0
        self.drop_unresolved_stop_sec = 3.0
        self.payload_unresolved = False
        self._drop_cmd_wall = None
        self.grip_dynamic_active = True
        self.hover_test_mode = False
        self.x_cur = np.array([
            10.0, -2.0, 6.0,       # position
            4.0, -1.0, 0.2,        # velocity at UNRESOLVED
            1.0, 0.0, 0.0, 0.0,    # attitude
            0.1, -0.2, 0.3])       # body rate
        self._unresolved_ref_t0 = None
        self._unresolved_ref_x0 = None
        self._unresolved_ref_p1 = None
        self._unresolved_ref_q1 = None
        self._traj_omega = np.ones((3, 2))
        self._traj_thrust = np.ones(2)
        self.ref_fn = lambda t: np.full(13, 99.0)
        self.ref_window_fn = lambda t, N, dt, z_hover=None: np.full((13, N+1), 99.0)
        self._log = _Logger()

    def get_clock(self):
        return _Clock(self)

    def get_logger(self):
        return self._log


