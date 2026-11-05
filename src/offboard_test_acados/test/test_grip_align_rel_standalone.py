#!/usr/bin/env python3
"""吸附前最后对准回归测试(不起 ROS)。"""

import time

import numpy as np

from offboard_test_acados import acados_nmpc_node as nn


class _Logger:
    def __init__(self):
        self.infos = []
        self.warns = []

    def info(self, msg):
        self.infos.append(msg)

    def warn(self, msg):
        self.warns.append(msg)


class _Stub:
    _update_grip_align = nn.AcadosNMPCNode._update_grip_align
    _align_err = nn.AcadosNMPCNode._align_err
    _align_ready = nn.AcadosNMPCNode._align_ready
    _target_rel_fresh = nn.AcadosNMPCNode._target_rel_fresh
    target_rel_cb = nn.AcadosNMPCNode.target_rel_cb

    def __init__(self, offset=(0.0, 0.0)):
        self.grip_align_offset = np.asarray(offset, dtype=float)
        self.grip_align_tau = 0.5
        self.grip_align_xy_tol = 0.012
        self.grip_align_wait_max = 4.0
        self.grip_enable_sent = False
        self.grip_x, self.grip_y = 1.0, 0.0
        self._target_rel = None
        self._target_rel_wall = None
        self._align_xy = None
        self._align_wait_start = None
        self._align_log_t = -1e9
        self.x_cur = np.zeros(13)
        self._log = _Logger()

    def get_logger(self):
        return self._log

    def feed(self, rel):
        class _M:
            pass
        m = _M()
        m.data = list(rel) + [-0.5]
        self.target_rel_cb(m)


def _settle(st, drone_xy, box_local_xy, n=200):
    st.x_cur[0:2] = drone_xy
    for k in range(n):
        st.feed(np.asarray(box_local_xy) - st.x_cur[0:2])
        st._update_grip_align(0.05 * k)


def test_target_converges_to_local_box_plus_offset():
    # 本地系里 box 实际在 (1.03, -0.02)(名义 1.0, 0.0 因 EKF 漂移而偏)
    st = _Stub(offset=(0.0, 0.05))
    _settle(st, drone_xy=[1.0, 0.0], box_local_xy=[1.03, -0.02])
    assert abs(st.grip_x - 1.03) < 1e-3
    assert abs(st.grip_y - 0.03) < 1e-3


def test_frozen_after_enable_and_not_updated_while_moving():
    st = _Stub()
    _settle(st, drone_xy=[1.0, 0.0], box_local_xy=[1.03, 0.0])
    x_before = st.grip_x
    st.grip_enable_sent = True
    _settle(st, drone_xy=[1.0, 0.0], box_local_xy=[1.10, 0.0], n=50)
    assert st.grip_x == x_before

    st2 = _Stub()
    st2.x_cur[3] = 0.5                    # 水平速度 > 0.3 m/s
    st2.feed([0.04, 0.0])
    st2._update_grip_align(0.0)
    assert st2.grip_x == 1.0 and st2._align_xy is None


def test_ready_gate_tolerance_and_timeout():
    st = _Stub(offset=(0.0, 0.0))
    st.feed([0.03, 0.0])                  # 对不准:3cm
    assert not st._align_ready(10.0)
    assert not st._align_ready(13.9)
    assert st._align_ready(14.0)          # 等满 4s 放行
    assert any('enabling anyway' in w for w in st._log.warns)

    st2 = _Stub(offset=(0.0, 0.05))
    st2.feed([0.0, -0.045])               # 机体 - box = (0, 0.045),误差 5mm
    assert st2._align_ready(0.0)



def test_stale_rel_does_not_count_as_aligned():
    st = _Stub()
    st.feed([0.0, 0.0])
    st._target_rel_wall = time.monotonic() - 1.0
    assert not st._align_ready(0.0)
