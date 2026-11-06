#!/usr/bin/env python3
"""mhe_node:速度+质量降阶 MHE 的 ROS2 节点(本工作空间唯一的估计器)。

输入(只有两路,与控制器的意图完全解耦):
  /mavros/local_position/odom             位置/机体速度/姿态/角速度
  <motor_speed_topic> (actuator_msgs)     电机转速 → T_phys = k_f·Σω²
输出:
  /mhe/payload_estimate        原子载荷帧(PayloadEstimate,NMPC 唯一的载荷输入)
  /acados_nmpc/mhe_mass_estimate  质量(仅健康时发布;主线 NMPC 不订阅)
  /acados_nmpc/tau_phys        电机反算体力矩(诊断)

与 ros2_ws_HJH 的 mhe_node(MHE_MODEL_VARIANT=reduced,2026-11-05 主线配置)相比,
NMPC 消费的字段(质量、惯量、空载置信度、健康位、解龄)逐帧一致(见
test_mhe_node_equivalence)。去掉的部分在主线下都不影响这些字段:14 状态模型、
一阶矩/偏心估计(帧里 s_xy、c_xy 恒为 0;主线 NMPC 偏心补偿关闭)、MHE 侧释放判据
(主线释放由 NMPC 的加速度计检测器 arm D 判)、意外脱落看门狗、遗忘因子、机动门控、
pre-offboard 冷启动。
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray
from actuator_msgs.msg import Actuators

from ..common import quat_to_rotmat
from ..lifecycle_trace import evt_line
from ..payload_estimate import (
    PayloadEstimate,
    inertia_from_mass,
    no_payload_evidence_score,
    relative_error_percent,
)
from ..residual_logger import ResidualLogger
from .estimator import MassHealthMonitor, ReducedMassMHE
from .params import p as P
from .solver_builder import ensure_solver
from .weights import M0_THETA, EventWeightSchedule

# --- 电机推力/力矩反算常数(x500 SDF;与 ros2_ws_HJH mhe_node 相同)---
MOTOR_CONSTANT = 8.54858e-06     # 单电机推力 = k_f·ω²
THRUST_CAL_GAIN = 1.0
ROTOR_X = np.array([0.174, -0.174, 0.174, -0.174])
ROTOR_Y = np.array([-0.174, 0.174, 0.174, -0.174])
MOMENT_CONSTANT = 0.016
YAW_DIR = -np.array([+1.0, +1.0, -1.0, -1.0])


class MHENode(Node):
    def __init__(self):
        super().__init__('mhe_node')
        log = self.get_logger()
        log.info(f'MHE node starting (reduced velocity-mass MHE, N={P.N}, '
                 f'm_B={P.m_B:.4f})')
        log.info(f'[mhe-variant] reduced (x=[v_world(3), m], p=[q_meas(4), T_phys]) '
                 f'| {P.describe()}')
        self.solver = ensure_solver()

        dp = self.declare_parameter
        gv = lambda n: self.get_parameter(n).value
        dp('motor_speed_topic', '/x500_payload_0/command/motor_speed')
        dp('residual_log_dir', '')
        dp('payload_input_fresh_sec', 0.30)
        dp('payload_solution_fresh_sec', 0.30)
        dp('payload_output_tau_sec', 0.25)
        dp('input_min_z', 0.5)                 # 已知输入的离地高度门控 [m]
        dp('motor_window_avg', True)
        dp('grip_payload_envelope', 0.3)
        dp('eval_true_payload_mass', 0.0)
        dp('no_payload_mass_full', 0.015)
        dp('no_payload_mass_zero', 0.060)
        dp('event_trigger_enable', True)
        dp('schedule_theta', [float(v) for v in M0_THETA])
        dp('confirm_thresh_alpha', 1.5)
        dp('resid_baseline_tau_sec', 3.0)
        dp('resid_persist_frames', 2)
        dp('resid_warmup_frames', 20)
        dp('resid_settle_tol_n', 0.8)
        dp('resid_confirm_enable', True)
        dp('payload_present_enter_mp', 0.09)
        dp('payload_present_enter_persist', 20)
        dp('payload_present_exit_mp', 0.02)
        dp('payload_present_exit_persist', 50)
        dp('payload_exit_steady_omega', 0.15)
        dp('payload_exit_steady_vel', 0.20)
        dp('thrust_map_fault_gain', 1.0)
        dp('motor_speed_noise_frac', 0.0)

        self.payload_input_fresh_sec = max(float(gv('payload_input_fresh_sec')), P.dt)
        self.payload_solution_fresh_sec = max(float(gv('payload_solution_fresh_sec')), P.dt)
        self.payload_output_tau = max(float(gv('payload_output_tau_sec')), 1e-3)
        self.input_min_z = float(gv('input_min_z'))
        self.motor_window_avg = bool(gv('motor_window_avg'))
        self.grip_payload_envelope = float(gv('grip_payload_envelope'))
        self.eval_true_payload_mass = float(gv('eval_true_payload_mass'))
        self._payload_conf_args = {
            'mass_full': float(gv('no_payload_mass_full')),
            'mass_zero': float(gv('no_payload_mass_zero')),
        }
        self.event_enabled = bool(gv('event_trigger_enable'))
        # 推力残差阈值 = alpha · g · 载荷包线(主线 1.5·9.81·0.3 ≈ 4.4 N)
        self.confirm_thresh = float(gv('confirm_thresh_alpha')) * P.g * self.grip_payload_envelope
        self.resid_ema_alpha = P.dt / max(float(gv('resid_baseline_tau_sec')), P.dt)
        self.resid_persist = max(1, int(gv('resid_persist_frames')))
        self.resid_warmup = max(1, int(gv('resid_warmup_frames')))
        self.resid_settle_tol = float(gv('resid_settle_tol_n'))
        self.resid_confirm_enable = bool(gv('resid_confirm_enable'))
        self.present_enter_mp = float(gv('payload_present_enter_mp'))
        self.present_enter_persist = max(1, int(gv('payload_present_enter_persist')))
        self.present_exit_mp = float(gv('payload_present_exit_mp'))
        self.present_exit_persist = max(1, int(gv('payload_present_exit_persist')))
        self.exit_steady_omega = float(gv('payload_exit_steady_omega'))
        self.exit_steady_vel = float(gv('payload_exit_steady_vel'))
        self.thrust_map_fault_gain = float(gv('thrust_map_fault_gain'))
        self.motor_speed_noise_frac = max(0.0, float(gv('motor_speed_noise_frac')))
        self._fault_rng = np.random.default_rng()
        if self.thrust_map_fault_gain != 1.0 or self.motor_speed_noise_frac > 0.0:
            log.warn(f'[thrust-fault] estimator-side thrust map gain='
                     f'{self.thrust_map_fault_gain:.3f}, motor-speed noise '
                     f'sigma={100 * self.motor_speed_noise_frac:.1f}% (plant/NMPC unchanged)')

        theta = np.array(gv('schedule_theta'), dtype=float)
        self.scheduler = EventWeightSchedule(theta)
        self.core = ReducedMassMHE(self.solver, P, scheduler=self.scheduler)
        self.health = MassHealthMonitor(P)
        log.info(f'weight schedule theta = [{", ".join(f"{v:.2f}" for v in theta)}], '
                 f'residual self-trigger: thresh={self.confirm_thresh:.1f}N '
                 f'persist={self.resid_persist} '
                 f'baseline_tau={P.dt / self.resid_ema_alpha:.1f}s '
                 f'(enabled={int(self.event_enabled and self.resid_confirm_enable)})')

        # --- 状态 ---
        self.x_meas = None
        self.thrust_phys = None
        self.tau_phys = None
        self.u_known = None
        self._motor_f_acc = np.zeros(4)
        self._motor_n = 0
        self._last_odom_rx_sec = None
        self._last_motor_rx_sec = None
        self._last_u_rx_sec = None
        self._last_solve_success_sec = None
        self._last_solve_ok = False
        self._inputs_were_fresh = False
        self.y_buf, self.u_buf = [], []
        self.frames = 0
        self.counter = 0
        self.m_est = P.m_B
        self._mass_ok = True
        self._mass_reason = ''
        self._in_transition = False
        self._solve_ms = []
        self._timing_dumped = 0
        self._m_payload_out = 0.0
        self._conf = 1.0
        # 残差自触发检测器
        self._resid_baseline = None
        self._resid_pending = 0
        self._resid_cross_frame = None
        self._resid_settled = False
        self._resid_stable = 0
        # 载荷存在状态机(只用于日志/追溯,不影响任何发布字段)
        self._present = False
        self._present_hi = 0
        self._present_lo = 0

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/mavros/local_position/odom',
                                 self.odom_cb, qos)
        motor_topic = gv('motor_speed_topic')
        self.create_subscription(Actuators, motor_topic, self.motor_speed_cb, 10)
        self.mass_pub = self.create_publisher(Float64, '/acados_nmpc/mhe_mass_estimate', 10)
        self.tau_phys_pub = self.create_publisher(Float64MultiArray, '/acados_nmpc/tau_phys', 10)
        self.payload_estimate_pub = self.create_publisher(
            Float64MultiArray, '/mhe/payload_estimate', 10)

        rl_dir = str(gv('residual_log_dir'))
        self.resid_log = ResidualLogger(rl_dir, time.strftime('%Y%m%d_%H%M%S'), meta={
            'MOTOR_CONSTANT': MOTOR_CONSTANT, 'THRUST_CAL_GAIN': THRUST_CAL_GAIN,
            'ROTOR_X': list(ROTOR_X), 'ROTOR_Y': list(ROTOR_Y),
            'MOMENT_CONSTANT': MOMENT_CONSTANT, 'estimator': 'reduced velocity-mass MHE',
            'motor_window_avg': int(self.motor_window_avg),
            'Jxx': P.Jxx, 'Jyy': P.Jyy, 'Jzz': P.Jzz, 'm_nominal': P.m_B, 'g': P.g,
            'mhe_N': P.N, 'mhe_dt': P.dt,
            'grip_payload_envelope': self.grip_payload_envelope,
            'eval_true_payload_mass': self.eval_true_payload_mass,
            'motor_speed_topic': motor_topic,
            'odom_frame': 'twist.linear/angular = body FLU (raw, not rotated to world)',
        })
        if self.resid_log.enabled:
            from sensor_msgs.msg import Imu
            self.create_subscription(Imu, '/mavros/imu/data',
                                     lambda m: self.resid_log.log_imu(m), qos)
            log.info(f'[resid] diagnostic logging -> {rl_dir} (with IMU)')

        self.timer = self.create_timer(P.dt, self.timer_cb)
        log.info('MHE node initialized; waiting for odometry + rotor speeds')

    # ------------------------------------------------------------ 时钟
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ 输入
    def odom_cb(self, msg):
        self.resid_log.log_odom(msg)
        pos, vel = msg.pose.pose.position, msg.twist.twist.linear
        q, om = msg.pose.pose.orientation, msg.twist.twist.angular
        qn = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        if qn < 1e-6:
            return
        qw, qx, qy, qz = q.w / qn, q.x / qn, q.y / qn, q.z / qn
        v_world = quat_to_rotmat(qw, qx, qy, qz) @ np.array([vel.x, vel.y, vel.z])
        x = np.array([pos.x, pos.y, pos.z, *v_world, qw, qx, qy, qz, om.x, om.y, om.z])
        if not np.all(np.isfinite(x)):
            return
        self.x_meas = x
        self._last_odom_rx_sec = self._now()

    def _input_gate_ok(self):
        """已知输入只在空中有效:推力 >= 1 N 且里程计高度 > input_min_z。
        地面支持力会让 T_phys 与 m·g 脱钩。⚠️ 抓取悬停高度(0.35 m)低于门槛,
        所以每次抓取前后约 6 s 估计器会清窗,离地后用窗口力平衡种子重起。"""
        if self.thrust_phys is None or not math.isfinite(self.thrust_phys) \
                or self.thrust_phys < P.seed_thrust_min:
            return False
        return self.x_meas is not None and self.x_meas[2] >= self.input_min_z

    def motor_speed_cb(self, msg):
        if len(msg.velocity) < 4:
            return
        w = np.array(msg.velocity[:4])
        if not np.all(np.isfinite(w)):
            return
        if self.motor_speed_noise_frac > 0.0:
            w = w * (1.0 + self.motor_speed_noise_frac * self._fault_rng.standard_normal(4))
        w2 = self.thrust_map_fault_gain * w * w
        f = THRUST_CAL_GAIN * MOTOR_CONSTANT * w2
        self.thrust_phys = float(THRUST_CAL_GAIN * MOTOR_CONSTANT * np.sum(w2))
        self._motor_f_acc += f
        self._motor_n += 1
        self.tau_phys = np.array([float(np.sum(ROTOR_Y * f)),
                                  float(-np.sum(ROTOR_X * f)),
                                  MOMENT_CONSTANT * float(np.sum(YAW_DIR * f))])
        now = self._now()
        self._last_motor_rx_sec = now
        self.resid_log.log_motor(msg, w, self.tau_phys, self.thrust_phys)
        if self._input_gate_ok():
            self.u_known = np.array([self.thrust_phys, *self.tau_phys])
            self._last_u_rx_sec = now

    def _drain_motor_avg(self):
        """上一采样区间内电机推力的平均(区间平均才是零阶保持的正确离散化)。"""
        if self._motor_n == 0:
            return None
        f = self._motor_f_acc / self._motor_n
        self._motor_f_acc = np.zeros(4)
        self._motor_n = 0
        return float(np.sum(f))

    # ------------------------------------------------------------ 健康
    def _inputs_fresh(self, now):
        return all(s is not None and 0.0 <= now - s <= self.payload_input_fresh_sec
                   for s in (self._last_odom_rx_sec, self._last_motor_rx_sec,
                             self._last_u_rx_sec))

    def _frame_health(self):
        now = self._now()
        age = (1.0e6 if self._last_solve_success_sec is None
               else max(0.0, now - self._last_solve_success_sec))
        healthy = (self._last_solve_ok and self._inputs_fresh(now)
                   and age <= self.payload_solution_fresh_sec and self._mass_ok)
        return bool(healthy), float(age)

    # ------------------------------------------------------------ 主循环
    def timer_cb(self):
        if self.x_meas is None or self.u_known is None:
            return
        if not self._inputs_fresh(self._now()):
            if self._inputs_were_fresh:
                self.y_buf.clear()
                self.u_buf.clear()
                self.core.reset_window()
                self._last_solve_ok = False
                self.get_logger().warn(
                    '[payload-health] input stream stale; MHE window cleared, '
                    'publishing held estimate as unhealthy')
            self._inputs_were_fresh = False
            self._publish_frame()
            return
        self._inputs_were_fresh = True
        self.resid_log.log_internal(self.m_est, False, None, self.thrust_phys,
                                    truth=self._truth_row())
        u_row = self.u_known.copy()
        T_avg = self._drain_motor_avg()
        if self.motor_window_avg and T_avg is not None:
            u_row[0] = T_avg
        self.y_buf.append(self.x_meas.copy())
        if len(self.y_buf) > P.N + 1:
            self.y_buf.pop(0)
        self.u_buf.append(u_row)
        if len(self.u_buf) > P.N:
            self.u_buf.pop(0)
        self.frames += 1
        self._update_presence()
        if self.event_enabled:
            self._residual_detect()
        if len(self.y_buf) == P.N + 1:
            self._solve_window()
        self._publish_frame()

    # ------------------------------------------------------------ 种子
    def _window_balance_mass(self):
        N = P.N
        if len(self.u_buf) < N or len(self.y_buf) < N + 1:
            return None
        y = np.asarray(self.y_buf[-(N + 1):], dtype=float)
        T = np.asarray([u[0] for u in self.u_buf[-N:]], dtype=float)
        if not np.all(np.isfinite(T)) or float(np.mean(T)) < P.seed_thrust_min:
            return None
        q = y[:N, 6:10]
        qn = np.linalg.norm(q, axis=1)
        if np.any(qn < 1e-6):
            return None
        q = q / qn[:, None]
        c = np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), 1e-3, 1.0)
        den = N * P.g + (y[N, 5] - y[0, 5]) / P.dt
        if not math.isfinite(den) or den < 0.5 * N * P.g:
            return None
        return float(np.sum(T * c) / den)

    def _seed_mass(self, why):
        """返回 (m_seed, from_thrust):窗口力平衡,退化时 T/g,再退化时 m_B。"""
        m = self._window_balance_mass()
        if m is not None:
            return float(np.clip(m, P.m_min_num, P.m_max)), True
        T = self.thrust_phys
        if T is None or not math.isfinite(T) or T < P.seed_thrust_min:
            self.get_logger().warn(f'mass seed ({why}): thrust unavailable, using m_B')
            return float(P.m_B), False
        return float(np.clip(T / P.g, P.m_min_num, P.m_max)), True

    # ------------------------------------------------------------ 求解
    def _solve_window(self):
        log = self.get_logger()
        core = self.core
        if core.x0_bar is None:
            m_seed, from_thrust = self._seed_mass('initial window')
            core.initialize(self.y_buf, m_seed)
            self.m_est = float(core.m_est)
            log.info(f'arrival prior seeded (reduced): m={core.x0_bar[3]:.3f} kg '
                     f'({"T_phys/g, no prior" if from_thrust else "m_nominal prior"})')
        try:
            core.load_window(self.y_buf, self.u_buf)
        except ValueError as exc:
            self._last_solve_ok = False
            log.warn(f'MHE window rejected (reduced): {exc}; holding m={self.m_est:.4f}')
            return
        in_tr = self.scheduler.apply(self.solver, self.frames - 1)
        if in_tr != self._in_transition:
            log.info('event transition ' + ('started: pre-event stages de-weighted'
                                            if in_tr else 'ended: nominal weights restored'))
            self._in_transition = in_tr

        res = core.solve_loaded(self.y_buf, seed_fn=self._seed_mass)
        self._solve_ms.append(res.solve_ms)
        if not res.ok:
            self._last_solve_ok = False
            log.warn(f'MHE solve failed [status={res.status}] sqp_iter={res.sqp_iter} '
                     f'{"(non-finite state) " if res.nonfinite else ""}'
                     f'variant=reduced, holding last valid m={self.m_est:.4f} kg '
                     f'(streak {res.fail_streak}, total {core.n_fail}/{core.n_solves}); '
                     'payload estimate marked unhealthy')
            if res.reinit:
                log.warn(f're-anchored arrival prior to current window after '
                         f'{core.reinit_after} consecutive failures (reduced; seed '
                         f'{res.reinit_seed:.3f} kg used only as initial guess, published '
                         f'm held at {self.m_est:.4f} kg; reinit #{core.n_reinit})')
            return

        self.m_est = float(res.m)
        self._last_solve_ok = True
        self._last_solve_success_sec = self._now()
        ok, reason, changed = self.health.update(self.m_est)
        self._mass_ok, self._mass_reason = ok, reason
        if changed and not ok:
            T = self.thrust_phys if self.thrust_phys is not None else float('nan')
            log.warn(f'[reduced-health] HEALTHY->UNHEALTHY reason={reason} '
                     f'm_est={self.m_est:.4f} band=[{P.health_low_kg:.4f},'
                     f'{P.health_high_kg:.4f}] lbx=[{P.m_min_num:.4f},{P.m_max:.4f}] '
                     f'T_phys={T:.2f}N -> payload estimate unhealthy, mass not forwarded to NMPC'
                     + (' (possible thrust-map UNDER-estimate)' if 'below' in reason else '')
                     + (' (possible thrust-map OVER-estimate / out of envelope)'
                        if 'above' in reason else ''))
        elif changed and ok:
            log.info(f'[reduced-health] UNHEALTHY->HEALTHY m_est={self.m_est:.4f} '
                     f'(in band for {P.health_recover_frames} frames; '
                     f'unhealthy frames total {self.health.unhealthy_frames})')
        if self._mass_ok:
            self.mass_pub.publish(Float64(data=self.m_est))
        if self.tau_phys is not None:
            self.tau_phys_pub.publish(Float64MultiArray(data=[float(v) for v in self.tau_phys]))
        self._log_periodic()
        if self.counter % 20 == 0:
            log.info(f'[reduced-health] healthy={int(self._mass_ok)} '
                     f'reason={self._mass_reason or "-"} m_est={self.m_est:.4f} '
                     f'unhealthy_frames={self.health.unhealthy_frames} '
                     f'solve_fail={core.n_fail}/{core.n_solves} reinit={core.n_reinit} '
                     f'last_sqp_iter={res.sqp_iter}')

    # ------------------------------------------------------------ 发布
    def _publish_frame(self):
        """一个原子载荷帧。健康时输出经一阶低通跟随估计;不健康时保持上一帧。"""
        healthy, age = self._frame_health()
        if healthy:
            a = 1.0 - math.exp(-P.dt / self.payload_output_tau)
            mp_target = max(float(self.m_est) - P.m_B, 0.0)
            self._m_payload_out += a * (mp_target - self._m_payload_out)
        m_out = P.m_B + self._m_payload_out
        s0 = np.zeros(2)       # 帧格式保留的一阶矩/偏心字段,主线恒为 0
        dJ_diag, J_diag = inertia_from_mass(m_out, P.m_B, [P.Jxx, P.Jyy, P.Jzz], P.rz_prior)
        if healthy:
            target = no_payload_evidence_score(self._m_payload_out, **self._payload_conf_args)
            a = 1.0 - math.exp(-P.dt / self.payload_output_tau)
            self._conf += a * (target - self._conf)
        est = PayloadEstimate(m_total=m_out, s_xy=s0, c_xy=s0, dJ_diag=dJ_diag,
                              J_diag=J_diag, no_payload_confidence=self._conf,
                              healthy=healthy, solution_age_sec=age)
        self.payload_estimate_pub.publish(Float64MultiArray(data=est.as_array().tolist()))

    # ------------------------------------------------------------ 事件降权
    def _residual_detect(self):
        """T_phys 相对慢基线的残差自触发:越阈并持续 persist 帧即通知降权调度器。
        纯从测量推力检测,不使用任何外部 attach/release 信号。"""
        T = self.thrust_phys
        if T is None:
            return
        if self.scheduler.event_frame is not None:
            self._resid_baseline = None
            self._resid_pending = 0
            self._resid_settled = False
            self._resid_stable = 0
            return
        if not self.resid_confirm_enable:
            return
        a = self.resid_ema_alpha
        if self._resid_baseline is None:
            self._resid_baseline = T
            self._resid_stable = 0
            return
        dev = abs(T - self._resid_baseline)
        if not self._resid_settled:
            self._resid_baseline = (1.0 - a) * self._resid_baseline + a * T
            if dev < self.resid_settle_tol:
                self._resid_stable += 1
                if self._resid_stable >= self.resid_warmup:
                    self._resid_settled = True
                    self.get_logger().info(
                        f'[no-signal] baseline settled at {self._resid_baseline:.2f}N '
                        f'(quiet {self.resid_warmup} frames); detector armed')
            else:
                self._resid_stable = 0
            return
        if dev > self.confirm_thresh:
            if self._resid_pending == 0:
                self._resid_cross_frame = self.frames - 1
            self._resid_pending += 1
            if self._resid_pending >= self.resid_persist:
                self.scheduler.notify_event(self._resid_cross_frame)
                self.get_logger().info(
                    f'[no-signal] mass mutation self-detected at frame '
                    f'{self._resid_cross_frame}: T_phys={T:.2f}N vs baseline '
                    f'{self._resid_baseline:.2f}N (dev {dev:.2f}N > {self.confirm_thresh:.1f}N '
                    f'over {self.resid_persist} frames, direction='
                    f'{"DROP" if T < self._resid_baseline else "ATTACH"}); '
                    'de-weighting pre-event stages')
                self._resid_pending = 0
            return
        self._resid_pending = 0
        self._resid_baseline = (1.0 - a) * self._resid_baseline + a * T

    # ------------------------------------------------------------ 载荷存在(日志)
    def _update_presence(self):
        prev = self._present
        m_p = float(self.m_est) - P.m_B
        if not self._present:
            self._present_hi = self._present_hi + 1 if m_p > self.present_enter_mp else 0
            if self._present_hi >= self.present_enter_persist:
                self._present, self._present_hi = True, 0
        else:
            quiet = (self.x_meas is not None
                     and float(np.linalg.norm(self.x_meas[10:13])) <= self.exit_steady_omega
                     and float(np.linalg.norm(self.x_meas[3:5])) <= self.exit_steady_vel)
            if not quiet:
                self._present_lo = 0
            elif m_p < self.present_exit_mp:
                self._present_lo += 1
                if self._present_lo >= self.present_exit_persist:
                    self._present, self._present_lo = False, 0
            else:
                self._present_lo = 0
        if self._present != prev:
            state = 'LOADED' if self._present else 'EMPTY'
            self.get_logger().info(f'[payload-state] {"EMPTY->LOADED" if self._present else "LOADED->EMPTY"} '
                                   f'(sustained m_p evidence, m_p={m_p:+.3f}kg)')
            self.get_logger().info(evt_line('declare', state=state,
                                            src='sustained m_p evidence', m_p=m_p))

    # ------------------------------------------------------------ 诊断
    def _truth_row(self):
        """评估专用真值行(只写日志)。sensor-only 下估计器不知道吸附状态,
        m_true 恒按空机写,与 ros2_ws_HJH 的日志口径一致。"""
        if self.eval_true_payload_mass <= 0.0:
            return None
        J = (P.Jxx, P.Jxx)
        return {'m_true': P.m_B, 'cx_hat': 0.0, 'cy_hat': 0.0, 'cx_true': 0.0,
                'cy_true': 0.0, 'dJ_hat': 0.0, 'dJ_true': 0.0, 'Jxx_hat': J[0],
                'Jxx_true': J[1], 'Jyz_hat': 0.0, 'Jyz_true': 0.0}

    def _log_periodic(self):
        self.counter += 1
        log = self.get_logger()
        if len(self._solve_ms) - self._timing_dumped >= 20:
            log.info('[mhe-timing] variant=reduced ms=[' + ','.join(
                f'{v:.3f}' for v in self._solve_ms[self._timing_dumped:]) + ']')
            self._timing_dumped = len(self._solve_ms)
        if self.counter % 20 != 0:
            return
        T = self.thrust_phys if self.thrust_phys is not None else float('nan')
        s = self._solve_ms[-20:]
        log.info(f'MHE mass estimate: {self.m_est:.3f} kg (T_phys={T:.2f}N) '
                 f'| solve={sum(s) / len(s):.1f}ms (max{max(s):.1f})')
        if self.eval_true_payload_mass > 0.0:
            log.info(f'[truth] m_hat={self.m_est:.4f} m_true={P.m_B:.4f} '
                     f'err={100 * (self.m_est - P.m_B) / P.m_B:+.2f}% (m_true = airframe; '
                     'estimator is not told about attach)')
        if self.tau_phys is not None:
            log.info(f'[tau_phys] roll={self.tau_phys[0]:+.3f} pitch={self.tau_phys[1]:+.3f} Nm')
        if self.thrust_phys is not None and self.x_meas is not None:
            q = self.x_meas[6:10] / max(float(np.linalg.norm(self.x_meas[6:10])), 1e-9)
            c = float(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), 1e-3, 1.0))
            mq = self.thrust_phys * c / P.g
            log.info(f'[mass-diag] m_est={self.m_est:.4f} T={self.thrust_phys:.3f}N '
                     f'tilt={math.degrees(math.acos(c)):.2f}deg cos={c:.5f} '
                     f'| T/g={self.thrust_phys / P.g:.4f} T*cos/g={mq:.4f} '
                     f'| m_est-T*cos/g={self.m_est - mq:+.4f}kg '
                     f'({relative_error_percent(self.m_est, mq):+.2f}%) '
                     f'vz={float(self.x_meas[5]):+.3f}')


def main():
    rclpy.init()
    node = MHENode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.resid_log.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
