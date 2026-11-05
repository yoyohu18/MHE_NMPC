#!/usr/bin/env python3
"""加速度计力平衡释放检测器(2026-11-01)。

独立于 MHE 的空载证据:m_acc = LPF(T_phys) / LPF(f_z)。比力 f_z=(T+D_z)/m 不含
重力,所以 m=T/f_z 不经过姿态、速度微分和转动模型 —— 与 MHE 的 m̂ 是两条独立的
观测路径。14 维档下 MHE 置信度的惯量分由 m̂ 推出,m̂ 在 4m/s 机动中周期塌陷
−11~−17% 时质量分、惯量分一起放行(final cohort B 臂带载误确认 114 次);这条
证据不随 m̂ 塌陷,也不读 MHE 的健康/新鲜度,MHE 失败帧照样有证据。

纯 Python、不依赖 ROS:NMPC 在电机/IMU 回调里喂数据,在控制周期里取置信度。
生命周期判决由 acados_nmpc_node 的双重门控负责:地面吸附期只计算；LIFT 后
连续有载 2s 锁存意外脱落看门狗，或由正常 release command 立即开启确认。
"""

import math

from .payload_estimate import acc_empty_score


class AccReleaseDetector:
    """m_acc 的低通、新鲜度判定和空载置信度低通。时间一律用调用方给的 ``now``(秒)。"""

    def __init__(self, m_body, mass_full=0.015, mass_zero=0.060,
                 mass_tau_sec=0.5, conf_tau_sec=0.25, fresh_sec=0.2):
        self.m_body = float(m_body)
        self.mass_full = float(mass_full)
        self.mass_zero = float(mass_zero)
        self.mass_tau = max(float(mass_tau_sec), 1e-3)
        self.conf_tau = max(float(conf_tau_sec), 1e-3)
        self.fresh_sec = float(fresh_sec)
        self.thrust = None
        self.motor_rx_sec = None
        self.imu_rx_sec = None
        self._T_lp = None
        self._f_lp = None
        self._lp_sec = None
        # 没有证据 ≠ 空载:置信度从 0 起,只有新鲜的 m_acc 才能把它推高。
        self.confidence = 0.0
        self._conf_sec = None

    def on_motor(self, thrust, now):
        if thrust is not None and math.isfinite(thrust):
            self.thrust = float(thrust)
            self.motor_rx_sec = float(now)

    def on_imu(self, f_z, now):
        """f_z:机体 FLU 比力 z 分量(悬停 ≈ +g)。推力取最近一次电机回调的值。"""
        now = float(now)
        self.imu_rx_sec = now
        if self.thrust is None or not math.isfinite(f_z):
            return
        if self._T_lp is None or self._lp_sec is None:
            self._T_lp, self._f_lp = self.thrust, float(f_z)
        else:
            a = 1.0 - math.exp(-max(now - self._lp_sec, 0.0) / self.mass_tau)
            self._T_lp += a * (self.thrust - self._T_lp)
            self._f_lp += a * (float(f_z) - self._f_lp)
        self._lp_sec = now

    def fresh(self, now):
        if self._T_lp is None:
            return False
        for t in (self.imu_rx_sec, self.motor_rx_sec):
            if t is None or not 0.0 <= now - t <= self.fresh_sec:
                return False
        return True

    def mass(self, now):
        """IMU 或电机不新鲜/未初始化时返回 None。f_z 下限 1 m/s² 防自由落体段除零。"""
        if not self.fresh(now):
            return None
        return self._T_lp / max(self._f_lp, 1.0)

    def update(self, now):
        """推进置信度低通并返回。证据不新鲜时保持原值 —— 持续计数由调用方按
        ``fresh`` 断开,与 MHE 档"不健康帧保持置信度、打断计数"同一约定。"""
        m = self.mass(now)
        if m is None:
            return self.confidence
        target = acc_empty_score(m, self.m_body, self.mass_full, self.mass_zero)
        if self._conf_sec is None:
            self.confidence = target
        else:
            a = 1.0 - math.exp(-max(now - self._conf_sec, 0.0) / self.conf_tau)
            self.confidence += a * (target - self.confidence)
        self._conf_sec = float(now)
        return self.confidence
