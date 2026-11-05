#!/usr/bin/env python3
"""平动残差补偿(L1 增广,标称质量取 m_est)回归测试(不起 ROS)。"""

import numpy as np

from offboard_test_acados import acados_nmpc_node as nn
from offboard_test_acados.acados_params import p
from offboard_test_acados.l1_adaptive import L1Augmentation


class _Stamp:
    def __init__(self, t):
        self.sec = int(t)
        self.nanosec = int(round((t - int(t)) * 1e9))


class _Odom:
    def __init__(self, t):
        class _H:
            pass
        self.header = _H()
        self.header.stamp = _Stamp(t)


class _Stub:
    _l1_update = nn.AcadosNMPCNode._l1_update
    _accept_empty = nn.AcadosNMPCNode._accept_empty

    def __init__(self, m_est):
        self.l1 = L1Augmentation(a_gain=10.0, omega_c=0.5)
        self.d_lumped = np.zeros(3)
        self._l1_last_stamp = None
        self._l1_log_count = 0
        self.m_est = m_est
        self._l1_T_phys = None
        self.payload_unresolved = False
        self.grip_drop_pending = False
        self.grip_drop_done = False
        self.grip_dropped = False
        self._acc_loss_watchdog_armed = False
        self.attach_time = None
        self.accept_confidence = 1.0
        self.no_payload_confidence = 1.0
        self.no_payload_conf_hold_frames = 20

    def get_logger(self):
        class _L:
            info = warn = staticmethod(lambda _m: None)
        return _L()


def _hover(st, m_true, seconds=20.0, dt=0.02):
    """真实质量 m_true 悬停平衡:推力 = m_true·g,实测速度恒 0。"""
    st._l1_T_phys = m_true * p.g
    t = 100.0
    for _ in range(int(seconds / dt)):
        st._l1_update(_Odom(t), np.eye(3), np.zeros(3))
        t += dt


def test_dhat_restores_true_hover_thrust_when_mest_is_low():
    m_true = p.m + 0.30
    m_est = p.m + 0.05                        # 14 维档 8 字中典型的低估
    st = _Stub(m_est)
    _hover(st, m_true)
    # NMPC 模型 vel_dot_z = T/m_est + d_z - g = 0  ->  T = m_est·(g - d_z)
    T_model = m_est * (p.g - st.d_lumped[2])
    assert abs(T_model - m_true * p.g) < 0.05
    assert abs(st.d_lumped[0]) < 1e-6 and abs(st.d_lumped[1]) < 1e-6


def test_dhat_zero_when_mest_is_right():
    st = _Stub(p.m + 0.30)
    _hover(st, p.m + 0.30)
    assert np.linalg.norm(st.d_lumped) < 1e-3



def test_uncommanded_accept_keeps_dhat_commanded_accept_resets():
    st = _Stub(p.m + 0.05)
    _hover(st, p.m + 0.30)
    d_before = st.d_lumped.copy()
    st._accept_empty()                        # 无投放指令(看门狗未武装):质量偏差还在,不清
    assert np.array_equal(st.d_lumped, d_before)

    st._no_payload_latched = False
    st.grip_drop_pending = True
    st._accept_empty()                        # 真投放确认:清零重估
    assert np.linalg.norm(st.d_lumped) == 0.0
