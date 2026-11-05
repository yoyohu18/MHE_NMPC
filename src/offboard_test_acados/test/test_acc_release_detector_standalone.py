#!/usr/bin/env python3
"""加速度计力平衡释放检测器(arm D,主线唯一的释放确认来源)回归测试(不起 ROS)。"""

from offboard_test_acados.acc_release_detector import AccReleaseDetector
from offboard_test_acados.mhe.params import p as mhe_p
from offboard_test_acados.payload_estimate import PayloadEstimate, acc_empty_score

from nmpc_stub import _NMPCStub

M_B = mhe_p.m_B
M_LOAD = M_B + 0.30
DT = 0.05            # NMPC 控制周期
SUB = 0.01           # 电机/IMU 回调间隔(100Hz,足够细)


def test_acc_empty_score_thresholds():
    assert acc_empty_score(None, M_B) == 0.0              # 没有证据 ≠ 空载
    assert acc_empty_score(float('nan'), M_B) == 0.0
    assert acc_empty_score(M_B + 0.005, M_B) == 1.0
    assert acc_empty_score(M_B - 0.05, M_B) == 1.0        # 比空机轻按空载(正部)
    assert acc_empty_score(M_B + 0.10, M_B) == 0.0        # 带 0.1kg 载荷:一票否决
    assert 0.0 < acc_empty_score(M_B + 0.0375, M_B) < 1.0


def _feed(det, m, f_z, t0, secs):
    """真实质量 m、比力 f_z 下喂电机推力 T=m·f_z 与 IMU,返回结束时刻。"""
    n = round(secs / SUB)
    for k in range(n):
        t = t0 + k * SUB
        det.on_motor(m * f_z, t)
        det.on_imu(f_z, t)
    return t0 + n * SUB


def test_detector_recovers_mass_through_maneuver_and_release():
    det = AccReleaseDetector(M_B)
    assert det.confidence == 0.0 and det.mass(0.0) is None
    t = _feed(det, M_LOAD, 10.6, 0.0, 3.0)       # 机动:比力 ≠ g,比值仍是质量
    assert abs(det.mass(t - SUB) - M_LOAD) < 1e-6
    assert det.update(t - SUB) == 0.0
    t = _feed(det, M_B, 9.81, t, 3.0)            # 释放:推力降到空机值
    assert abs(det.mass(t - SUB) - M_B) < 0.01
    assert det.update(t - SUB) > 0.999            # 置信度一阶低通,一步 3s 后已到顶


def test_detector_holds_confidence_and_reports_stale():
    det = AccReleaseDetector(M_B)
    t = _feed(det, M_B, 9.81, 0.0, 1.0)
    assert det.update(t - SUB) == 1.0
    late = t + 0.5                                # IMU/电机 0.5s 没来
    assert not det.fresh(late) and det.mass(late) is None
    assert det.update(late) == 1.0                # 保持,由调用方按 fresh 断开计数
    det.on_motor(M_LOAD * 9.81, late)             # 只有电机恢复也不算新鲜
    assert not det.fresh(late + 0.01)


class _AccStub(_NMPCStub):
    def __init__(self):
        super().__init__()
        self._acc_det = AccReleaseDetector(M_B)
        self._acc_log_n = 0
        self.accept_confidence = 1.0
        self.m_est = M_B
        self.acc_loss_arm_hold_frames = round(2.0 / DT)
        self._acc_loss_watchdog_armed = False
        self._acc_loaded_arm_frames = 0
        self.attach_time = 0.0
        self.grip_lift_started = True
        self.grip_mass_stepped = True
        self.nmpc_start_time = None
        self.dJ_est = 0.1
        self.grip_drop_pending = True
        self._drop_cmd_wall = self.now_sec


def _run(st, m, secs, f_z=9.81, mhe_conf=1.0, mhe_healthy=True, sensors=True):
    """每个控制周期:先喂一个周期的传感器,再设 MHE 那一路,再跑确认链。"""
    for _ in range(round(secs / DT)):
        if sensors:
            _feed(st._acc_det, m, f_z, st.now_sec, DT)
        st.now_sec += DT
        st.no_payload_confidence = mhe_conf
        st._payload_estimate_rx_sec = st.now_sec
        st._payload_target = PayloadEstimate(
            m_total=M_B, s_xy=[0.0, 0.0], c_xy=[0.0, 0.0], dJ_diag=[0.0] * 3,
            J_diag=[1.0] * 3, no_payload_confidence=mhe_conf,
            healthy=mhe_healthy, solution_age_sec=0.0)
        st._confirm_no_payload_if_persistent()
        if st.grip_dropped:
            return True
    return False


def test_acc_vetoes_mhe_collapse_while_loaded():
    """m̂ 塌到空机、MHE 说空(conf=1),但加速度计说有载:不确认。"""
    st = _AccStub()
    assert not _run(st, M_LOAD, 5.0, f_z=10.6, mhe_conf=1.0)
    assert st.accept_confidence < 0.01


def test_acc_confirms_without_healthy_mhe():
    """指令直接开 gate；即使未走 LIFT 且 MHE 不健康,acc 仍能独立确认。"""
    st = _AccStub()
    st.grip_lift_started = False
    _run(st, M_LOAD, 2.0, mhe_conf=0.0, mhe_healthy=False)
    t0 = st.now_sec
    assert _run(st, M_B, 6.0, mhe_conf=0.0, mhe_healthy=False)
    assert 1.0 <= st.now_sec - t0 < 3.0



def test_stale_sensors_break_persistence():
    st = _AccStub()
    _run(st, M_LOAD, 2.0)
    _run(st, M_B, 0.8)                            # 已攒 ~0.6s,不够 1s
    assert not st.grip_dropped
    assert not _run(st, M_B, 0.5, sensors=False)  # IMU/电机断流:计数清零
    assert st._no_payload_conf_frames == 0
    t0 = st.now_sec
    assert _run(st, M_B, 3.0)
    assert st.now_sec - t0 >= 1.0                 # 恢复后重新攒满 1s


def test_acc_rearms_latch_on_fresh_loaded_evidence():
    st = _AccStub()
    st._no_payload_latched = True
    st.grip_drop_pending = False
    _run(st, M_LOAD, 1.0)
    assert st._no_payload_latched is False


def test_ground_pickup_transient_cannot_confirm_before_airborne_arm():
    """地面支撑下 acc 可先说有载再说空,但 gate 未开不得消费生命周期事件。"""
    st = _AccStub()
    st.grip_drop_pending = False
    st.grip_lift_started = False
    _run(st, M_LOAD, 1.0)       # 短暂载荷转移,足以解除旧 latch
    assert not _run(st, M_B, 4.0)
    assert not st._acc_loss_watchdog_armed
    assert not st.grip_dropped


def test_airborne_loaded_dwell_arms_and_detects_uncommanded_loss():
    """LIFT 后连续2s有载才锁存;之后没有 release command 也能确认意外脱落。"""
    st = _AccStub()
    st.grip_drop_pending = False
    # 小于 dwell 的假有载证据不能武装；随后假空载也不能确认。
    for _ in range(st.acc_loss_arm_hold_frames - 1):
        st._update_acc_loss_watchdog(0.0, True)
    assert not st._acc_loss_watchdog_armed
    st._update_acc_loss_watchdog(1.0, True)  # 连续性断开,计数必须归零
    assert not _run(st, M_B, 3.0)
    # 真正稳定带载后 watchdog 永久武装。
    _run(st, M_LOAD, 3.0)
    assert st._acc_loss_watchdog_armed
    # 无命令脱落:acc 变空并持续,走 payload-lost 安全复位。
    assert _run(st, M_B, 4.0)
    assert st.grip_dropped and st.grip_drop_done
    assert st.dJ_est == 0.0
    assert st.l1.resets >= 1                      # 意外脱落复位 L1
