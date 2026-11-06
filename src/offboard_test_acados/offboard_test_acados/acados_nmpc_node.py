#!/usr/bin/env python3
# acados NMPC 节点(主线:磁吸夹爪吊运 + 降阶 MHE 载荷自适应 + 加速度计力平衡释放确认)。
#
# 状态机:预热 → 等 EKF → 切 OFFBOARD → 解锁 → 起飞点垂直爬升 → 平移到 box 上方 →
# NMPC 接管 → 稳定 → 下降并最后对准 → 发 enable 吸附 → 抬升 → 8 字 → 在 8 字尖端
# 投放 → 加速度计证据持续确认释放(arm D)→ 继续飞行。
#
# 2026-11-05 自 ros2_ws_HJH 的同名节点精简而来:按主线参数常量折叠,删去主线走不到的
# 分支(旧接口/真值几何/online 几何、L1 与 fixed 控制基线、旧释放基线、MHE 侧释放
# 确认、偏心补偿、c_xy 冻结、离地武装、LIFT hold、PX4 增益缩放、非夹爪模式等)。
# 主线行为与旧节点逐位等价(离线对拍:全部发布输出与运行期日志逐位相同),需要旧
# 分支时回到 ros2_ws_HJH。

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64, Float64MultiArray
from actuator_msgs.msg import Actuators

from .common import quat_to_rotmat
from .figure8_reference import (
    auto_ramp_time,
    build_reference_figure8,
    build_reference_window_figure8,
)
from .acados_params import (
    OMEGA_MIN as ACT_OMEGA_MIN,
    OMEGA_SPAN as ACT_OMEGA_SPAN,
    THROTTLE_MAX as ACT_THROTTLE_MAX,
    THROTTLE_MIN as ACT_THROTTLE_MIN,
    THRUST_K as ACT_THRUST_K,
    p,
)
from .acados_solver_builder import ensure_acados_ocp_solver
from .l1_adaptive import L1Augmentation
from .lifecycle_trace import evt_line
from .payload_estimate import PayloadEstimate, headroom_limited_scale, slew
from .acc_release_detector import AccReleaseDetector
from .mhe.params import p as mhe_p

# 转子标定常数。**必须与 mhe 侧的同名常数保持一致**:从电机转速反算实测推力
# T_phys = MOTOR_CONSTANT·Σω²(L1 残差与加速度计释放检测器都用它)。
MOTOR_CONSTANT = 8.54858e-06
MOTOR_OMEGA_MAX = 1000.0   # gz x500 model.sdf 的 maxRotVelocity(rad/s),仅诊断用


class AcadosNMPCNode(Node):
    def __init__(self):
        super().__init__('acados_nmpc_node')
        self.get_logger().info('acados NMPC node starting, building/loading solver...')

        self.solver = ensure_acados_ocp_solver()

        self.state      = State()
        self.x_cur      = None
        self.counter    = 0
        self.start_time = None
        self.armed_and_flying  = False
        self.last_mode_req_time = 0.0
        self.last_arm_req_time  = 0.0
        self.ekf_wait_sec = 10.0
        self.nmpc_started = False
        self.nmpc_start_time = None
        # 归一化油门 <-> 真实推力的物理映射(PX4 offboard thrust 直通电机,
        # GZMixingInterfaceESC: ω = OMEGA_MIN + OMEGA_SPAN*norm,电机 T = THRUST_K*ω²)。
        # 三个常数与 NMPC 箱式约束 p.Tmin/p.Tmax 同源(acados_params)。反解在 publish_attitude。
        self.OMEGA_MIN  = ACT_OMEGA_MIN
        self.OMEGA_SPAN = ACT_OMEGA_SPAN
        self.THRUST_K   = ACT_THRUST_K
        self.omega_cmd_max = 2.0
        self.bodyrate_ramp_time = 0.8
        # True = 参考时间冻结在 0(固定悬停);进入 8 字 / unresolved 制动时置 False。
        self.hover_test_mode = True

        # 求解失败处理:偶尔失败一次不扰动 warm-start;连续失败约 2s 才拉回悬停猜测。
        self.solve_fail_count = 0
        self.max_consecutive_fail = max(1, round(2.0 / p.dt))
        self.last_u_opt = p.u_hover.copy()
        self.last_omega_cmd = np.zeros(3)

        # 解耦发布:NMPC 以 1/dt Hz 求解整条预测轨迹,高频定时器沿轨迹插值前推发
        # body_rate + thrust。_traj_omega (3,N+1) 未 ramp 的角速度轨迹,_traj_thrust (N,)
        # 推力轨迹;None = 退化为零阶保持。_traj_stamp 用 ROS clock(插值网格是仿真时间)。
        self._traj_omega = None
        self._traj_thrust = None
        self._traj_stamp = None
        self._omega_grid = np.arange(p.N + 1) * p.dt
        self._thrust_grid = np.arange(p.N) * p.dt

        self.max_res_stat = 0.0
        self.last_res_stat = 0.0
        self.last_sqp_iter = 0

        # ---- 连续载荷估计:/mhe/payload_estimate 是 NMPC 唯一的载荷认知输入 ----
        # 每次求解前对 MHE 帧做低通 + 限速,得到模型用的 m_est / dJ_est。
        self.declare_parameter('payload_model_tau_sec', 0.20)
        self.declare_parameter('payload_mass_slew_kg_s', 0.60)
        self.declare_parameter('payload_dj_slew_kgm2_s', 0.080)
        self.declare_parameter('no_payload_confidence_threshold', 0.90)
        self.declare_parameter('no_payload_confidence_hold_sec', 1.0)
        self.declare_parameter('payload_estimate_fresh_sec', 0.35)
        self.payload_model_tau = max(
            float(self.get_parameter('payload_model_tau_sec').value), 1e-3)
        self.payload_mass_slew = float(
            self.get_parameter('payload_mass_slew_kg_s').value)
        self.payload_dj_slew = float(
            self.get_parameter('payload_dj_slew_kgm2_s').value)
        self.no_payload_conf_threshold = float(
            self.get_parameter('no_payload_confidence_threshold').value)
        self.no_payload_conf_hold_frames = max(1, round(
            float(self.get_parameter('no_payload_confidence_hold_sec').value) / p.dt))
        self.payload_estimate_fresh_sec = max(
            float(self.get_parameter('payload_estimate_fresh_sec').value), p.dt)
        self._no_payload_conf_frames = 0
        self._no_payload_latched = True
        self._payload_estimate_received = False
        self._payload_estimate_rx_sec = None
        self._payload_estimate_health_prev = None
        self._payload_target = PayloadEstimate(
            m_total=p.m, s_xy=np.zeros(2), c_xy=np.zeros(2),
            dJ_diag=np.zeros(3), J_diag=np.array([p.Jxx, p.Jyy, p.Jzz]),
            no_payload_confidence=1.0)
        # MHE 发布的空载置信度:attach J bootstrap 与日志读它;释放确认链不读它。
        self.no_payload_confidence = 1.0

        # ---- 释放确认:加速度计力平衡检测器(arm D)----
        # m_acc = LPF(T_phys)/LPF(f_z),本节点自订 IMU,T_phys 来自电机转速回调。不经过
        # MHE,持续计数只看 IMU/电机新鲜度。确认链置信度 accept_confidence = acc_conf。
        # 双重启用门控:正常投放由释放指令开启确认链;无指令时只有"LIFT 已开始且 acc
        # 连续给出有载证据"后锁存的意外脱落看门狗能开启。吸附/地面阶段只计算、不确认。
        self.declare_parameter('payload_output_tau_sec', 0.25)
        self.declare_parameter('no_payload_mass_full', 0.015)
        self.declare_parameter('no_payload_mass_zero', 0.060)
        self.declare_parameter('acc_mass_tau_sec', 0.5)
        self.declare_parameter('acc_fresh_sec', 0.2)
        self.declare_parameter('acc_loss_arm_hold_sec', 2.0)
        self.declare_parameter('imu_topic', '/mavros/imu/data')
        self.accept_confidence = 1.0
        self._acc_log_n = 0
        self.acc_loss_arm_hold_frames = max(1, round(float(
            self.get_parameter('acc_loss_arm_hold_sec').value) / p.dt))
        self._acc_loss_watchdog_armed = False
        self._acc_loaded_arm_frames = 0
        self._acc_det = AccReleaseDetector(
            mhe_p.m_B,
            mass_full=float(self.get_parameter('no_payload_mass_full').value),
            mass_zero=float(self.get_parameter('no_payload_mass_zero').value),
            mass_tau_sec=float(self.get_parameter('acc_mass_tau_sec').value),
            conf_tau_sec=float(self.get_parameter('payload_output_tau_sec').value),
            fresh_sec=float(self.get_parameter('acc_fresh_sec').value))

        self.m_est = p.m       # 空机起飞
        # 吊挂载荷惯量增量(model.p 几何槽第 1 维)。偏心补偿关:几何槽恒为 [dJ, 0, 0]。
        self.dJ_est = 0.0
        # 平动 lumped 扰动比力(model.p 最后 3 维),由 L1Augmentation 在线估计。
        self.d_lumped = np.zeros(3)

        # ---- 估计器侧推力映射失配注入(标定误差测试)----
        # 只缩放加速度计释放检测器读到的 T;L1 残差与执行器油门映射不动。与 mhe_node
        # 同名参数同源(run_sitl.sh 的 THRUST_FAULT_GAIN 同时喂两边)。
        self.declare_parameter('thrust_map_fault_gain', 1.0)
        self.thrust_map_fault_gain = float(
            self.get_parameter('thrust_map_fault_gain').value)

        # ---- 平动残差补偿(L1 增广,标称质量取 m_est)----
        # 模型无积分作用,m̂ 偏差会直接变成推力前馈误差;d̂ 只补"模型 vs 实测"的残差。
        # a_nom 的推力项用电机转速反算的实际推力(不能用 NMPC 指令推力:会自举炸机)。
        # 复位只在真投放(确认)与意外脱落时做。
        self.declare_parameter('l1_a_gain', 10.0)    # 预测器带宽
        self.declare_parameter('l1_omega_c', 0.5)    # 补偿低通截止(1.0/2.0 实测与外环耦合致 z 振荡)
        self.l1 = L1Augmentation(
            a_gain=float(self.get_parameter('l1_a_gain').value),
            omega_c=float(self.get_parameter('l1_omega_c').value))
        self._l1_last_stamp = None
        self._l1_log_count = 0

        # ---- ω_cmd 缩放:增益调度的等效实现 ----
        # 发给 PX4 的是 body-rate setpoint,速率环 τ = K(ω_cmd − ω)。真实惯量 Ĵ_t 下把
        # setpoint 误差放大 Ĵ_t/J_a:ω_cmd' = ω + s·(ω_cmd − ω),s = (Jxx+dJ_est)/Jxx,
        # 只作用 roll/pitch,完全在本进程内,不写 PX4 参数。D 项是对测量微分,不跟着涨。
        self.declare_parameter('omega_scale_tau', 0.5)
        self.declare_parameter('omega_scale_cap', 5.0)
        self.declare_parameter('omega_scale_hysteresis', 0.08)
        self.declare_parameter('omega_scale_rise_rate', 4.0)
        self.declare_parameter('omega_scale_recover_rate', 1.5)
        self.declare_parameter('omega_scale_headroom_fraction', 0.95)
        self.omega_scale_tau = float(self.get_parameter('omega_scale_tau').value)
        self.omega_scale_cap = float(self.get_parameter('omega_scale_cap').value)
        self.omega_scale_hysteresis = float(
            self.get_parameter('omega_scale_hysteresis').value)
        self.omega_scale_rise_rate = float(
            self.get_parameter('omega_scale_rise_rate').value)
        self.omega_scale_recover_rate = float(
            self.get_parameter('omega_scale_recover_rate').value)
        self.omega_scale_headroom_fraction = float(np.clip(
            self.get_parameter('omega_scale_headroom_fraction').value,
            0.5, 1.0))
        self.omega_scale = 1.0
        self._omega_scale_target = 1.0
        self._omega_scale_stamp = None
        self._omega_scale_clip_warned = False

        # ---- 飞行安全诊断(只读,每 5s 一条 [flight-diag])----
        self._fd_mot_peak = 0.0
        self._fd_mot_sat = 0
        self._fd_mot_n = 0
        self._fd_reset()
        self.get_logger().info(
            f'OMEGA SCALE: on (source=djest, '
            f'tau={self.omega_scale_tau:.2f}s, cap={self.omega_scale_cap:.1f}) '
            '— 增益调度走 setpoint 侧,PX4 参数不动')

        self._l1_T_phys = None      # None = 还没收到转速,L1 不推进
        self.declare_parameter('motor_speed_topic', '/x500_0/command/motor_speed')
        self.create_subscription(
            Actuators, self.get_parameter('motor_speed_topic').value,
            self._l1_motor_cb, 10)
        self.get_logger().info(
            f'LUMPED FORCE: on (a={self.l1.a:.1f}, omega_c={self.l1.omega_c:.1f})'
            ' — L1 平动残差补偿,标称质量取 m_est')
        if self.thrust_map_fault_gain != 1.0:
            self.get_logger().warn(
                f'[thrust-fault] estimator-side thrust map gain='
                f'{self.thrust_map_fault_gain:.3f} (acc detector only; '
                'L1 residual and actuator throttle map unchanged)')

        # ---- 投放未证实(UNRESOLVED)----
        # 指令后超时仍无释放证据:不清模型、不复位 L1,用有限时长平滑制动退出机动。
        self.declare_parameter('drop_unresolved_timeout_sec', 12.0)
        self.drop_unresolved_timeout_sec = float(
            self.get_parameter('drop_unresolved_timeout_sec').value)
        self.declare_parameter('drop_unresolved_stop_sec', 3.0)
        self.drop_unresolved_stop_sec = max(0.5, float(
            self.get_parameter('drop_unresolved_stop_sec').value))
        self.payload_unresolved = False
        self._drop_cmd_wall = None
        self._unresolved_ref_t0 = None
        self._unresolved_ref_x0 = None
        self._unresolved_ref_p1 = None
        self._unresolved_ref_q1 = None

        # ============ 夹爪任务参数(默认值即主线取值)============
        self.declare_parameter('grip_x', 1.0)
        self.declare_parameter('grip_y', 0.0)
        self.declare_parameter('grip_z_low', 0.35)   # 吸附时低空悬停高度
        self.declare_parameter('grip_z_high', 6.0)   # 抬升后悬停高度
        self.declare_parameter('grip_lift_after_sec', 3.0)
        self.declare_parameter('grip_lift_dur', 8.4)
        # 投放:抬升完成后至少 grip_drop_after_sec 秒,且精确落在 8 字最左端(a=3π/2)。
        self.declare_parameter('grip_drop_after_sec', 45.0)
        # 带载 8 字:r=10 m;SPEED=2 → w=0.1415/ramp=9.36,SPEED=4 → w=0.283/ramp=0。
        self.declare_parameter('grip_dyn_r', 10.0)
        self.declare_parameter('grip_dyn_w', 0.1415)
        # ramp_time:0 = 写死的 4.0;<0 = auto_ramp_time(grip_dyn_w);>0 = 显式指定。
        self.declare_parameter('grip_dyn_ramp', 9.36)
        # 立体 8 字高度起伏 z = grip_z_high + dz·sin(a);投放点 a=3π/2 是轨迹最低点。
        self.declare_parameter('grip_dyn_dz', 0.8)
        self.declare_parameter('grip_dyn_settle_sec', 3.0)  # lift 完成后多久切 8 字
        # 意外脱落故障注入:在计划投放的同一相位走 /gripper/release 物理脱离,不发释放指令。
        self.declare_parameter('grip_uncommanded_loss', False)
        # 吊挂力臂 d[m]:attach J bootstrap 地板 dJ = μ·d² 用。
        self.declare_parameter('grip_arm_d', 0.27)
        # attach J bootstrap:载荷尚由地面承重时惯量不可观,先给 J/dJ 一个机架包线地板
        # (m 与 s_xy 仍完全来自 MHE),地板保持到投放指令,之后在 release_sec 内平滑交给
        # MHE 的 dJ。地板只来自载荷**包线上界**(机架规格),不含任务信息。
        self.declare_parameter('attach_j_bootstrap_release_sec', 0.5)
        self.declare_parameter('grip_payload_envelope', 0.3)
        # 两段式接近:先在安全高度 grip_approach_z 对齐 box 并由 NMPC 接管,稳定
        # grip_settle_sec(给 MHE 窗口热身)后用 grip_descend_dur 平滑下降到 grip_z_low。
        self.declare_parameter('grip_approach_z', 1.5)
        self.declare_parameter('grip_settle_sec', 3.0)
        self.declare_parameter('grip_descend_dur', 2.0)
        # 斜坡走完后还要等真实高度/速度收敛才发 enable,避免机体滞后时过早吸附。
        self.declare_parameter('grip_descend_z_tol', 0.02)
        self.declare_parameter('grip_descend_v_tol', 0.15)
        # 最后对准:用 proximity 发的"box - 机体"相对位置把悬停水平目标改成本地系的
        # box + offset(EKF 本地系相对 gz 世界系每轮漂 2~4cm)。只用于导航。
        self.declare_parameter('grip_align_offset_x', 0.0)
        self.declare_parameter('grip_align_offset_y', 0.0)
        self.declare_parameter('grip_align_tau_sec', 0.5)
        self.declare_parameter('grip_align_xy_tol', 0.012)
        self.declare_parameter('grip_align_wait_max_sec', 4.0)
        # [attach-window] 逐帧诊断日志的窗口时长
        self.declare_parameter('attach_window_sec', 140.0)

        self.grip_x = float(self.get_parameter('grip_x').value)
        self.grip_y = float(self.get_parameter('grip_y').value)
        self.grip_z_low = float(self.get_parameter('grip_z_low').value)
        self.grip_z_high = float(self.get_parameter('grip_z_high').value)
        self.grip_lift_after_sec = float(
            self.get_parameter('grip_lift_after_sec').value)
        self.grip_lift_dur = float(self.get_parameter('grip_lift_dur').value)
        self.grip_drop_after_sec = float(
            self.get_parameter('grip_drop_after_sec').value)
        self.grip_dyn_r = float(self.get_parameter('grip_dyn_r').value)
        self.grip_dyn_w = float(self.get_parameter('grip_dyn_w').value)
        _gr = float(self.get_parameter('grip_dyn_ramp').value)
        self.grip_dyn_ramp = (auto_ramp_time(self.grip_dyn_w) if _gr < 0.0
                              else (4.0 if _gr == 0.0 else _gr))
        self.grip_dyn_dz = float(self.get_parameter('grip_dyn_dz').value)
        self.grip_dyn_settle_sec = float(
            self.get_parameter('grip_dyn_settle_sec').value)
        self.grip_uncommanded_loss = bool(
            self.get_parameter('grip_uncommanded_loss').value)
        self.grip_arm_d = float(self.get_parameter('grip_arm_d').value)
        self.grip_payload_envelope = float(
            self.get_parameter('grip_payload_envelope').value)
        self.attach_j_bootstrap_release_sec = max(
            float(self.get_parameter(
                'attach_j_bootstrap_release_sec').value), p.dt)
        self._attach_j_bootstrap_active = False
        self._attach_j_bootstrap_releasing = False
        self._attach_j_bootstrap_weight = 0.0
        self._attach_j_bootstrap_dj = 0.0
        self.grip_approach_z = float(self.get_parameter('grip_approach_z').value)
        self.grip_settle_sec = float(self.get_parameter('grip_settle_sec').value)
        self.grip_descend_dur = float(self.get_parameter('grip_descend_dur').value)
        self.grip_descend_z_tol = float(
            self.get_parameter('grip_descend_z_tol').value)
        self.grip_descend_v_tol = float(
            self.get_parameter('grip_descend_v_tol').value)
        self.grip_align_offset = np.array([
            float(self.get_parameter('grip_align_offset_x').value),
            float(self.get_parameter('grip_align_offset_y').value)])
        self.grip_align_tau = max(
            1e-3, float(self.get_parameter('grip_align_tau_sec').value))
        self.grip_align_xy_tol = float(self.get_parameter('grip_align_xy_tol').value)
        self.grip_align_wait_max = float(
            self.get_parameter('grip_align_wait_max_sec').value)
        self.attach_window_sec = float(self.get_parameter('attach_window_sec').value)
        self._target_rel = None          # 最近一次 [dx,dy](box - 机体)
        self._target_rel_wall = None
        self._align_xy = None            # 滤波后的本地系悬停目标
        self._align_wait_start = None
        self._align_log_t = -1e9
        self.get_logger().info(
            f'[align] 最后对准开:目标 = 本地系 box + offset '
            f'{np.array2string(self.grip_align_offset, precision=3)}, '
            f'tol={self.grip_align_xy_tol:.3f}m, '
            f'wait_max={self.grip_align_wait_max:.1f}s')

        # 任务状态
        self.grip_climbed = False        # 接近第 0 段:起飞点原地爬到安全高度
        self.grip_high_aligned = False   # 接近第 1 段:安全高度对齐 box
        self.grip_descend_started = False
        self.grip_descend_done = False
        self.grip_descend_done_time = None  # descend 完成(发 enable)的 nmpc_time
        self.grip_enable_sent = False
        self.attach_time = None          # attach 指令的 nmpc_time(lift 时序起点)
        self.grip_mass_stepped = False   # attach 已处理
        self.grip_lift_started = False
        self.grip_drop_done = False      # 投放已触发/已完成(只一次)
        self.grip_drop_pending = False   # 已发释放指令,等待确认
        self.grip_dropped = False        # 已确认载荷脱离
        self.grip_dynamic_active = False  # 当前是否在飞 8 字(unresolved 制动会清掉)
        # "只切一次"的哨兵必须与"当前是否在机动"分开:unresolved 会清 grip_dynamic_active,
        # 若拿它当哨兵,下一拍 8 字会被原地重启(要求 4m/s 的飞机瞬间刹停 → 穿地,5/5 复现)。
        self._grip_dyn_latched = False
        self.grip_dyn_t0 = None          # 8 字起始 nmpc_time
        self.grip_dyn_cx = 0.0           # 8 字 xy 偏移(切换瞬间 drone 位置)
        self.grip_dyn_cy = 0.0
        # NMPC 主动门控 attach:proximity 收到 /gripper/enable=true 才判定几何。
        self.enable_pub = self.create_publisher(Bool, '/gripper/enable', 10)
        self.grip_release_pub = self.create_publisher(Bool, '/gripper/release', 10)
        # NMPC 在安全高度接管(给 MHE 窗口攒事件前数据)
        self.z_hover = self.grip_approach_z
        self.ref_fn = self._grip_ref
        self.ref_window_fn = self._grip_ref_window
        # 到达接管点的判据(低空地效 + 位置控制稳态下垂,放宽)
        self.start_xy_threshold = 0.30
        self.start_z_threshold = 0.35
        self.get_logger().info(
            f'GRIPPER MODE: hover over box ({self.grip_x},{self.grip_y}) '
            f'z {self.grip_z_low}->{self.grip_z_high}m, lift after '
            f'{self.grip_lift_after_sec}s, empty m_est={self.m_est:.2f}kg, '
            f'attach J bootstrap=True '
            f'({self.grip_payload_envelope:.2f}kg envelope)')

        xref0 = self.ref_fn(0.0)
        # 预热求解器:把远离收敛点时的 SQP 迭代代价提前消化,避免发生在接管那一帧。
        self.get_logger().info('Warming up acados solver...')
        t_warm = time.time()
        self._seed_initial_guess(xref0, np.tile(xref0.reshape(-1, 1), (1, p.N + 1)))
        for _ in range(10):
            u_opt, omega_cmd, _ = self.solve_nmpc(xref0, 0.0)
        self.get_logger().info(
            f'Solver warmup complete, took {(time.time()-t_warm)*1000:.1f} ms')

        mavros_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        mavros_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_cb, mavros_state_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/local_position/odom',
            self.odom_cb, mavros_sensor_qos)
        # 唯一的载荷认知输入。attach/drop 通知、夹爪真值和 payload_lost 都不进 NMPC。
        self.payload_estimate_sub = self.create_subscription(
            Float64MultiArray, '/mhe/payload_estimate',
            self.payload_estimate_cb, 10)
        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value,
            self._acc_imu_cb, mavros_sensor_qos)
        d = self._acc_det
        self.get_logger().info(
            f'[acc-det] release_confirm_source=acc: '
            f'm_acc=LPF(T_phys)/LPF(f_z), tau={d.mass_tau:.2f}s, '
            f'conf_tau={d.conf_tau:.2f}s, fresh<={d.fresh_sec:.2f}s, '
            f'阈值 {d.mass_full:.3f}/{d.mass_zero:.3f}kg, m_B={d.m_body:.4f}')
        self.target_rel_sub = self.create_subscription(
            Float64MultiArray, '/gripper/target_rel', self.target_rel_cb, 1)

        self.att_pub = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', 10)
        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)
        # 话题名保留 acados_ 前缀,与既有日志和分析工具兼容。
        self.ref_path_pub = self.create_publisher(
            Path, '/acados_nmpc/reference_path', 10)
        self.actual_path_pub = self.create_publisher(
            Path, '/acados_nmpc/actual_path', 10)
        self.nmpc_traj_pub = self.create_publisher(
            Path, '/acados_nmpc_traj', 10)
        self.tracking_err_pub = self.create_publisher(
            Float64, '/acados_nmpc/tracking_error', 10)
        # 实际采用的 [T,taux,tauy,tauz](诊断;MHE 以自己的 10Hz 消费最新值)
        self.u_opt_pub = self.create_publisher(
            Float64MultiArray, '/acados_nmpc/u_opt', 10)

        self.ref_path_msg = self._build_ref_path_msg()
        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = 'map'
        self.nmpc_traj_msg = Path()
        self.nmpc_traj_msg.header.frame_id = 'map'
        self.actual_path_max_len = 2000

        self.viz_timer = self.create_timer(0.1, self._publish_paths)

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.timer = self.create_timer(p.dt, self.timer_cb)

        self.declare_parameter('publish_hz', 50.0)
        publish_hz = float(self.get_parameter('publish_hz').value)
        self.pub_timer = self.create_timer(1.0 / publish_hz, self._publish_from_traj)
        self.get_logger().info(
            f'Decoupled publish enabled: solve @{1.0/p.dt:.0f}Hz, '
            f'body_rate/thrust @{publish_hz:.0f}Hz (trajectory interpolation)')
        self.get_logger().info('acados NMPC node initialized! Waiting for EKF2 convergence...')

    # ------------------------------------------------------------------ 回调
    def state_cb(self, msg):
        self.state = msg

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        q   = msg.pose.pose.orientation
        om  = msg.twist.twist.angular
        qn = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        if qn < 1e-6:
            return
        qw, qx, qy, qz = q.w/qn, q.x/qn, q.y/qn, q.z/qn
        # body(FLU) -> world(ENU)
        R_wb = quat_to_rotmat(qw, qx, qy, qz)
        vel_world = R_wb @ np.array([vel.x, vel.y, vel.z])
        x = np.array([
            pos.x, pos.y, pos.z,
            vel_world[0], vel_world[1], vel_world[2],
            qw, qx, qy, qz,
            om.x, om.y, om.z
        ])
        if not np.all(np.isfinite(x)):
            return
        self.x_cur = x
        self._l1_update(msg, R_wb, vel_world)
        self._append_actual_path(msg)

    def _l1_motor_cb(self, msg):
        # 口径与 mhe_node 同源:T_phys = k_f·Σω²(4 电机)。
        if len(msg.velocity) >= 4:
            w = np.asarray(msg.velocity[:4], dtype=float)
            if np.all(np.isfinite(w)):
                wn = float(np.max(np.abs(w))) / MOTOR_OMEGA_MAX
                self._fd_mot_peak = max(self._fd_mot_peak, wn)
                self._fd_mot_n += 1
                if wn >= 0.98:
                    self._fd_mot_sat += 1
                w2 = w * w
                self._l1_T_phys = float(MOTOR_CONSTANT * np.sum(w2))
                self._acc_det.on_motor(
                    self.thrust_map_fault_gain * self._l1_T_phys,
                    self.get_clock().now().nanoseconds * 1e-9)

    def _acc_imu_cb(self, msg):
        self._acc_det.on_imu(float(msg.linear_acceleration.z),
                             self.get_clock().now().nanoseconds * 1e-9)

    def _l1_update(self, odom_msg, R_wb, vel_world):
        """L1 增广的高频更新(odom 频率)。a_nom 按模型 vel_dot 用 m_est、d=0 算,
        推力用电机转速反算的 T_phys;dt 用 odom 时间戳差。"""
        if self._l1_T_phys is None:     # 还没收到转速:不推进
            return
        # 地面门控:地面支持力会被 L1 估成 d_z=+g。低推力时清零挂起,离地重新估计。
        if self._l1_T_phys < 0.4 * p.m * p.g:
            if np.any(self.d_lumped != 0.0):
                self.get_logger().info('[l1] low-thrust gate: reset (on ground)')
            self.l1.reset()
            self.d_lumped = np.zeros(3)
            self._l1_last_stamp = None
            return
        t = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        if self._l1_last_stamp is None:
            self._l1_last_stamp = t
            return
        dt = t - self._l1_last_stamp
        self._l1_last_stamp = t
        if not (0.0 < dt < 0.5):        # 时基跳变/暂停帧:跳过
            return
        m_nom = max(float(self.m_est), 1e-3)
        a_nom = (R_wb @ np.array([0.0, 0.0, self._l1_T_phys])
                 - p.kd * vel_world) / m_nom - np.array([0.0, 0.0, p.g])
        self.d_lumped = self.l1.update(vel_world, a_nom, dt)
        self._l1_log_count += 1
        if self._l1_log_count % 100 == 0:
            d = self.d_lumped
            self.get_logger().info(
                f'[l1] d_hat_f=[{d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}]m/s^2 '
                f'|equiv_dm={-d[2] * p.m / p.g:+.3f}kg')

    def payload_estimate_cb(self, msg):
        """Receive the sole MHE->NMPC payload state (one coherent MHE frame)."""
        try:
            estimate = PayloadEstimate.from_array(msg.data)
        except ValueError as exc:
            self.get_logger().warn(f'ignoring invalid payload estimate: {exc}')
            return
        self._payload_target = estimate
        self._payload_estimate_received = True
        self._payload_estimate_rx_sec = (
            self.get_clock().now().nanoseconds * 1e-9)
        self.no_payload_confidence = estimate.no_payload_confidence

    # ------------------------------------------------------------ 释放确认链
    def _update_acc_confirm(self):
        """推进加速度计检测器,得到确认链置信度,并按新鲜的有载证据重新武装。
        每个控制周期调一次(在持续计数之前)。"""
        now = self.get_clock().now().nanoseconds * 1e-9
        det = self._acc_det
        acc_conf = det.update(now)
        acc_fresh = det.fresh(now)
        self._update_acc_loss_watchdog(acc_conf, acc_fresh)
        self.accept_confidence = acc_conf
        # 重新武装:只认新鲜的"有载"证据,阈值 thr−0.15。
        rearm_below = max(0.0, self.no_payload_conf_threshold - 0.15)
        if acc_fresh and acc_conf < rearm_below:
            self._no_payload_latched = False
            self._no_payload_conf_frames = 0
        self._acc_log_n += 1
        if self._acc_log_n % 10 == 0:
            m_acc = det.mass(now)
            self.get_logger().info(
                f'[acc-det] m_acc={m_acc if m_acc is not None else float("nan"):.4f} '
                f'acc_conf={acc_conf:.3f} fresh={int(acc_fresh)} '
                f'mhe_conf={self.no_payload_confidence:.3f} '
                f'accept={self.accept_confidence:.3f} '
                f'loss_armed={int(self._acc_loss_watchdog_armed)} '
                f'latched={int(self._no_payload_latched)} '
                f'm_hat={float(self.m_est):.4f}')

    def _reset_acc_loss_watchdog(self):
        """A new pickup starts with release confirmation inhibited."""
        self._acc_loss_watchdog_armed = False
        self._acc_loaded_arm_frames = 0

    def _update_acc_loss_watchdog(self, acc_conf, acc_fresh):
        """Arm uncommanded-loss detection only after stable airborne load.

        ``grip_lift_started`` is the deployable airborne gate (controller-owned
        mission state, not Gazebo truth).  Requiring fresh
        ``acc_conf < threshold-0.15`` for a dwell proves the vehicle is actually
        carrying load, not merely connected to a box still supported by the
        ground.  Once armed the latch is never cleared by rising empty
        confidence -- that rise is precisely the loss event to detect.
        """
        if self._acc_loss_watchdog_armed:
            return
        airborne_gate = self.attach_time is not None and self.grip_lift_started
        loaded = (airborne_gate and acc_fresh
                  and acc_conf < max(
                      0.0, self.no_payload_conf_threshold - 0.15))
        if not loaded:
            self._acc_loaded_arm_frames = 0
            return
        self._acc_loaded_arm_frames += 1
        if self._acc_loaded_arm_frames < self.acc_loss_arm_hold_frames:
            return
        self._acc_loss_watchdog_armed = True
        self._acc_loaded_arm_frames = 0
        self.get_logger().info(evt_line(
            'acc_loss_armed', conf=float(acc_conf),
            hold_s=self.acc_loss_arm_hold_frames * p.dt))
        self.get_logger().info(
            f'[acc-det] airborne loaded evidence persisted for '
            f'{self.acc_loss_arm_hold_frames * p.dt:.2f}s; '
            'uncommanded-loss confirmation ARMED')

    def _confirm_no_payload_if_persistent(self):
        """空载证据持续 hold 秒(且新鲜、连续)才接受载荷已脱离。"""
        self._update_acc_confirm()
        self._check_drop_unresolved()
        if self._no_payload_latched:
            return
        # 双重门控:释放指令边沿立即开启;无指令时只有已武装的意外脱落看门狗能开启。
        if not (self.grip_drop_pending or self._acc_loss_watchdog_armed):
            self._no_payload_conf_frames = 0
            return
        if not self._acc_det.fresh(self.get_clock().now().nanoseconds * 1e-9):
            # 持续计数必须在有效证据时间上连续:陈旧帧打断而不是暂停计数。
            self._no_payload_conf_frames = 0
            return
        if self.accept_confidence >= self.no_payload_conf_threshold:
            self._no_payload_conf_frames += 1
        else:
            self._no_payload_conf_frames = 0
            return
        if self._no_payload_conf_frames < self.no_payload_conf_hold_frames:
            return
        self._no_payload_latched = True
        self._no_payload_conf_frames = 0
        self.get_logger().info(evt_line(
            'confirm', conf=float(self.accept_confidence),
            hold_s=self.no_payload_conf_hold_frames * p.dt,
            pending=bool(self.grip_drop_pending)))
        self._accept_empty()

    def _accept_empty(self):
        """控制器接受"载荷已脱离"。"""
        _pending = bool(self.grip_drop_pending)
        self._no_payload_latched = True
        self._no_payload_conf_frames = 0
        self.get_logger().info(evt_line(
            'accept', src='confidence', pending=_pending,
            unresolved=bool(self.payload_unresolved)))
        if self.payload_unresolved:
            self.payload_unresolved = False
            self.get_logger().info(
                'PAYLOAD STATE RESOLVED: 迟到的 moment 证据到达,恢复正常卸载流程')
            self.get_logger().info(evt_line('resolved'))
        if _pending:
            # 真投放确认:模型质量此刻切到空机,之前补质量偏差的 d̂ 作废。
            self.l1.reset()
            self.d_lumped = np.zeros(3)
            self._l1_last_stamp = None
            self.grip_drop_pending = False
            self.grip_drop_done = True
            self.grip_dropped = True
            self.get_logger().info(
                f'DROP complete: acc no-payload confidence '
                f'{self.accept_confidence:.3f} persisted for '
                f'{self.no_payload_conf_hold_frames * p.dt:.2f}s; '
                'adaptive states reset')
        elif self._acc_loss_watchdog_armed and self.attach_time is not None:
            # 没有释放指令而 airborne-loaded 看门狗已武装:这是意外脱落。
            self._handle_uncommanded_payload_loss('acc confidence')

    def _handle_uncommanded_payload_loss(self, source):
        """意外脱落(不是自己松的夹爪):归零模型惯量 + 复位 L1。
        不发 enable=false(夹爪状态未知)。幂等。"""
        if not self.grip_mass_stepped or self.grip_dropped:
            return
        self.grip_dropped = True
        self.grip_drop_done = True      # 抑制之后的计划投放(货已经没了)
        self.dJ_est = 0.0
        self.l1.reset()
        self.d_lumped = np.zeros(3)
        # 日志格式对齐阶段行 `t=..s | XXX`,供论文图/演示视频解析器标出事件。
        _t = (0.0 if self.nmpc_start_time is None else
              (self.get_clock().now() - self.nmpc_start_time).nanoseconds / 1e9)
        self.get_logger().warn(
            f't={_t:.1f}s | LOST: PAYLOAD LOST ({source}) 非计划脱落 → '
            '内环增益已复位、模型几何归零')

    def _check_drop_unresolved(self):
        """释放指令后超时仍未确认 → unresolved:既不清模型也不猜,退出机动转悬停。

        猜"已卸载"而载荷其实还在 = 带载飞空机构型;继续按带载飞而载荷真掉了 =
        带着幽灵载荷飞。两种都危险,所以保持模型不动、平滑制动,交给上层/操作员。
        """
        if not self.grip_drop_pending:
            return
        if self.payload_unresolved:
            return
        if self._drop_cmd_wall is None:
            return
        if (self.get_clock().now().nanoseconds * 1e-9 - self._drop_cmd_wall
                < self.drop_unresolved_timeout_sec):
            return
        self.payload_unresolved = True
        hover_armed = self._begin_unresolved_hover()
        action = (f'用 {self.drop_unresolved_stop_sec:.1f}s 平滑制动后在停止点悬停'
                  if hover_armed else
                  '当前状态无效，未能切换制动参考，保持原参考并要求立即上层接管')
        self.get_logger().error(
            f'PAYLOAD STATE UNRESOLVED: drop 指令后 '
            f'{self.drop_unresolved_timeout_sec:.0f}s 未取得 moment 证据。'
            f'**不清模型、不切 s_target、不复位 L1**,{action}。'
            f'(conf={self.no_payload_confidence:.3f}) '
            f'—— 载荷是否仍在机上未知,需上层介入/降落中止')
        self.get_logger().info(evt_line(
            'unresolved', conf=float(self.no_payload_confidence),
            timeout_s=self.drop_unresolved_timeout_sec))
        self.get_logger().info(evt_line(
            'brake', armed=bool(hover_armed), stop_s=self.drop_unresolved_stop_sec))

    def _begin_unresolved_hover(self):
        """Replace the dynamic reference with a continuous stop-to-hover path.

        Payload/model state is deliberately untouched.  The reference starts from the
        measured state, decelerates with zero acceleration at both endpoints, then remains
        stationary, avoiding the position/velocity step of switching straight to hover.
        """
        x0 = np.asarray(self.x_cur, dtype=float).copy()
        if (x0.shape != (13,) or not np.all(np.isfinite(x0))
                or np.linalg.norm(x0[6:10]) < 1e-6):
            self.get_logger().error(
                'UNRESOLVED hover not armed: current state/quaternion is unavailable or invalid')
            return False
        self.grip_dynamic_active = False
        self.hover_test_mode = False  # the braking reference needs advancing NMPC time
        self._unresolved_ref_x0 = x0
        T = self.drop_unresolved_stop_sec
        # A half-duration velocity projection is the endpoint of a symmetric smooth stop.
        self._unresolved_ref_p1 = x0[0:3] + 0.5 * T * x0[3:6]

        q0 = x0[6:10] / max(np.linalg.norm(x0[6:10]), 1e-9)
        # Level attitude preserving current yaw, same hemisphere as q0.
        qw, qx, qy, qz = q0
        yaw = np.arctan2(2.0 * (qw*qz + qx*qy),
                         1.0 - 2.0 * (qy*qy + qz*qz))
        q1 = np.array([np.cos(0.5*yaw), 0.0, 0.0, np.sin(0.5*yaw)])
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        self._unresolved_ref_x0[6:10] = q0
        self._unresolved_ref_q1 = q1
        # t0 is latched by the first query, in the same t_ref clock used by solve_nmpc.
        self._unresolved_ref_t0 = None
        self.ref_fn = self._grip_unresolved_ref
        self.ref_window_fn = self._grip_unresolved_ref_window
        # Drop the old figure-eight prediction; until the next solve the publisher
        # zero-order-holds, preserving the PX4 offboard stream.
        self._traj_omega = None
        self._traj_thrust = None
        return True

    def _grip_unresolved_ref(self, t=0.0, **kw):
        """Finite-time smooth braking followed by a stationary hover reference."""
        if self._unresolved_ref_t0 is None:
            self._unresolved_ref_t0 = float(t)
        tau = np.clip(float(t) - self._unresolved_ref_t0,
                      0.0, self.drop_unresolved_stop_sec)
        T = self.drop_unresolved_stop_sec
        u = tau / T
        x0 = self._unresolved_ref_x0
        p0, v0, p1 = x0[0:3], x0[3:6], self._unresolved_ref_p1

        # Quintic Hermite: p/v continuous, zero acceleration at both ends.
        dp = p1 - p0
        a3 = (10.0*dp - 6.0*T*v0) / T**3
        a4 = (-15.0*dp + 8.0*T*v0) / T**4
        a5 = (6.0*dp - 3.0*T*v0) / T**5
        pos = p0 + v0*tau + a3*tau**3 + a4*tau**4 + a5*tau**5
        vel = v0 + 3.0*a3*tau**2 + 4.0*a4*tau**3 + 5.0*a5*tau**4

        blend = 10.0*u**3 - 15.0*u**4 + 6.0*u**5
        q = (1.0 - blend) * x0[6:10] + blend * self._unresolved_ref_q1
        q /= max(np.linalg.norm(q), 1e-9)
        omega = (1.0 - blend) * x0[10:13]
        return np.concatenate([pos, vel, q, omega])

    def _grip_unresolved_ref_window(self, t_start, N, dt, z_hover=None):
        return np.column_stack([
            self._grip_unresolved_ref(t_start + i*dt) for i in range(N + 1)])

    # ------------------------------------------------------------ 载荷模型
    def _payload_estimate_is_fresh(self):
        """True only for a healthy MHE solution and a fresh ROS delivery."""
        if (not self._payload_estimate_received
                or self._payload_estimate_rx_sec is None
                or not self._payload_target.healthy):
            fresh = False
            rx_age = float('inf')
        else:
            now = self.get_clock().now().nanoseconds * 1e-9
            rx_age = max(0.0, now - self._payload_estimate_rx_sec)
            fresh = (rx_age <= self.payload_estimate_fresh_sec
                     and self._payload_target.solution_age_sec
                     <= self.payload_estimate_fresh_sec)
        if (self._payload_estimate_received
                and fresh != self._payload_estimate_health_prev):
            # ⚠️ rclpy 按调用点缓存 severity:同一行先 info 后 warn 会抛 ValueError,
            # 在定时器回调里会直接带走节点。两个分支必须是两个物理调用点。
            if fresh:
                self.get_logger().info(
                    '[payload-health] healthy/fresh; model updates enabled')
            else:
                self.get_logger().warn(
                    f'[payload-health] unhealthy/stale; holding model and '
                    f'confidence timer '
                    f'(mhe_healthy={self._payload_target.healthy}, '
                    f'solution_age={self._payload_target.solution_age_sec:.3f}s, '
                    f'rx_age={rx_age:.3f}s)')
            self._payload_estimate_health_prev = fresh
        return fresh

    def _start_attach_j_bootstrap(self):
        """Arm the conservative roll/pitch-inertia floor for an attach attempt.
        地板值只来自 grip_payload_envelope(机架档位先验),不读任何估计。"""
        self._attach_j_bootstrap_dj = self._dJ_from_mp(
            max(0.0, self.grip_payload_envelope))
        self._attach_j_bootstrap_weight = 1.0
        self._attach_j_bootstrap_releasing = False
        self._attach_j_bootstrap_active = self._attach_j_bootstrap_dj > 0.0
        if self._attach_j_bootstrap_active:
            self.get_logger().info(
                'ATTACH J bootstrap armed: '
                f'm_p,envelope={self.grip_payload_envelope:.3f}kg, '
                f'dJxx=dJyy={self._attach_j_bootstrap_dj:.4f}kg m^2, '
                f'Jxx={p.Jxx + self._attach_j_bootstrap_dj:.4f}, '
                f'Jyy={p.Jyy + self._attach_j_bootstrap_dj:.4f}; '
                'm/s_xy remain MHE-only')

    def _update_continuous_payload_model(self):
        """Smooth and slew-limit the MHE state before every NMPC solve."""
        bootstrap_active = self._attach_j_bootstrap_active
        estimate_fresh = self._payload_estimate_is_fresh()
        if not estimate_fresh and not bootstrap_active:
            return
        dt = p.dt
        alpha = 1.0 - math.exp(-dt / self.payload_model_tau)
        t = self._payload_target
        if estimate_fresh:
            m_target = float(np.clip(t.m_total, p.m, mhe_p.m_max))
            m_lp = self.m_est + alpha * (m_target - self.m_est)
            self.m_est = float(slew(
                self.m_est, m_lp, self.payload_mass_slew, dt))
            dj_mhe = max(0.0, float(np.mean(t.dJ_diag[:2])))
        else:
            # No MHE frame: m is held, while J can still reach its safe pre-lift value.
            dj_mhe = 0.0

        # Once a handoff has started, loss of estimator validity freezes it at the
        # last safe inertia (releasing toward zero would recreate the LIFT failure).
        if (bootstrap_active and self._attach_j_bootstrap_releasing
                and not estimate_fresh):
            return

        if bootstrap_active:
            # 地板保持到投放指令:投放后 MHE 已看不到载荷,由指令直接开始交接。
            if (not self._attach_j_bootstrap_releasing
                    and (self.grip_drop_pending or self.grip_dropped)):
                self._attach_j_bootstrap_releasing = True
                self.get_logger().info(
                    'ATTACH J bootstrap held until drop: drop command '
                    f'issued, blending floor {self._attach_j_bootstrap_dj:.4f} '
                    f'-> MHE dJ {dj_mhe:.4f}kg m^2 over '
                    f'{self.attach_j_bootstrap_release_sec:.2f}s')
            if self._attach_j_bootstrap_releasing:
                self._attach_j_bootstrap_weight = max(
                    0.0, self._attach_j_bootstrap_weight
                    - dt / self.attach_j_bootstrap_release_sec)
                if self._attach_j_bootstrap_weight <= 0.0:
                    self._attach_j_bootstrap_active = False
                    bootstrap_active = False
                    self.get_logger().info(
                        'ATTACH J bootstrap released; NMPC J is now MHE-only')

        # The overlay changes J only; LPF + slew make rise and handoff continuous.
        if bootstrap_active:
            floor_extra = max(self._attach_j_bootstrap_dj - dj_mhe, 0.0)
            dj_target = dj_mhe + self._attach_j_bootstrap_weight * floor_extra
        else:
            dj_target = dj_mhe
        dj_lp = self.dJ_est + alpha * (dj_target - self.dJ_est)
        self.dJ_est = float(slew(
            self.dJ_est, dj_lp, self.payload_dj_slew, dt))

    def _dJ_from_mp(self, m_p):
        """吊挂惯量增量 dJ = μ·d²(μ = 约化质量 m_b·m_p/m_t)。"""
        m_t = p.m + m_p
        mu = p.m * m_p / m_t if m_t > 0 else 0.0
        return float(mu * self.grip_arm_d ** 2)

    # ------------------------------------------------------------ 可视化
    def _build_ref_path_msg(self, z_hover=3.0, n_samples=400, hover_time=2.0,
                            w=0.3, ramp_time=4.0, t_offset=0.0):
        # 从 hover_time+ramp_time 之后采样一整周期(振幅已渐变完)。形状由 ref_fn 决定;
        # w / ramp_time / t_offset 必须与实际生效的那组一致,否则画不满一圈。
        msg = Path()
        msg.header.frame_id = 'map'
        period = 2.0 * math.pi / w
        t0 = t_offset + hover_time + ramp_time
        for i in range(n_samples + 1):
            t = t0 + i * period / n_samples
            xref = self.ref_fn(t, z_hover=z_hover)
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(xref[0])
            ps.pose.position.y = float(xref[1])
            ps.pose.position.z = float(xref[2])
            ps.pose.orientation.w = float(xref[6])
            ps.pose.orientation.x = float(xref[7])
            ps.pose.orientation.y = float(xref[8])
            ps.pose.orientation.z = float(xref[9])
            msg.poses.append(ps)
        return msg

    def _publish_paths(self):
        now = self.get_clock().now().to_msg()
        self.ref_path_msg.header.stamp = now
        self.ref_path_pub.publish(self.ref_path_msg)
        self.actual_path_msg.header.stamp = now
        self.actual_path_pub.publish(self.actual_path_msg)
        self.nmpc_traj_msg.header.stamp = now
        self.nmpc_traj_pub.publish(self.nmpc_traj_msg)

    def _append_actual_path(self, odom_msg):
        ps = PoseStamped()
        ps.header.stamp = odom_msg.header.stamp
        ps.header.frame_id = 'map'
        ps.pose = odom_msg.pose.pose
        self.actual_path_msg.poses.append(ps)
        if len(self.actual_path_msg.poses) > self.actual_path_max_len:
            self.actual_path_msg.poses.pop(0)

    def _set_nmpc_traj_msg(self, X_sol):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i in range(X_sol.shape[1]):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(X_sol[0, i])
            ps.pose.position.y = float(X_sol[1, i])
            ps.pose.position.z = float(X_sol[2, i])
            ps.pose.orientation.w = float(X_sol[6, i])
            ps.pose.orientation.x = float(X_sol[7, i])
            ps.pose.orientation.y = float(X_sol[8, i])
            ps.pose.orientation.z = float(X_sol[9, i])
            msg.poses.append(ps)
        self.nmpc_traj_msg = msg

    def pub_hover_pos(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = self.z_hover
        pose.pose.orientation.w = 1.0
        self.pos_pub.publish(pose)

    def pub_position_ref(self, xref):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(xref[0])
        pose.pose.position.y = float(xref[1])
        pose.pose.position.z = float(xref[2])
        pose.pose.orientation.w = float(xref[6])
        pose.pose.orientation.x = float(xref[7])
        pose.pose.orientation.y = float(xref[8])
        pose.pose.orientation.z = float(xref[9])
        self.pos_pub.publish(pose)

    # ------------------------------------------------------------ 求解与发布
    def _seed_initial_guess(self, x_init, Xref_win):
        # warm-start 猜测的悬停推力按当前 m_est 算(固定 p.m 在满载时会让 SQP 卡进病态)。
        u_hover_now = np.array([self.m_est * p.g, 0.0, 0.0, 0.0])
        for i in range(p.N):
            self.solver.set(i, 'x', Xref_win[:, i])
            self.solver.set(i, 'u', u_hover_now)
        self.solver.set(p.N, 'x', Xref_win[:, p.N])

    def _att_err_deg(self):
        """实测姿态与上一次成功求解的一步预测姿态之间的夹角 [deg](日志诊断)。"""
        q1 = getattr(self, '_q_pred1', None)
        if q1 is None or self.x_cur is None:
            return float('nan')
        q0 = np.asarray(self.x_cur[6:10], dtype=float)
        n = np.linalg.norm(q0) * np.linalg.norm(q1)
        if not n > 0.0:
            return float('nan')
        c = min(1.0, abs(float(np.dot(q0, q1))) / n)
        return math.degrees(2.0 * math.acos(c))

    def solve_nmpc(self, x_cur, t_ref):
        # 确认逻辑可能在 UNRESOLVED 超时时替换 ref_fn/ref_window_fn,故必须先判决、
        # 再构造本周期的参考窗口。
        self._update_continuous_payload_model()
        self._confirm_no_payload_if_persistent()

        Xref_win = self.ref_window_fn(t_ref, p.N, p.dt, z_hover=self.z_hover)

        self.solver.set(0, 'lbx', x_cur)
        self.solver.set(0, 'ubx', x_cur)
        geom_slot = np.array([self.dJ_est, 0.0, 0.0])   # [dJ, cx, cy],偏心补偿关
        for i in range(p.N + 1):
            self.solver.set(i, 'p', np.concatenate(
                [Xref_win[:, i], [self.m_est], geom_slot, self.d_lumped]))

        # t_start_ros:轨迹 t=0 对应的 ROS 时刻(高频发布算 τ 用);t_start:墙钟,只测耗时。
        t_start_ros = self.get_clock().now()
        t_start = time.time()
        status = self.solver.solve()

        # 诊断:KKT 残差和 SQP 迭代次数(不管成功失败都记录)
        res_stat, res_eq, res_ineq, res_comp = self.solver.get_stats('residuals')
        sqp_iter = self.solver.get_stats('sqp_iter')
        self.max_res_stat = max(self.max_res_stat, float(res_stat))
        self.last_res_stat = float(res_stat)
        self.last_sqp_iter = int(sqp_iter)

        if res_stat > 10.0 or sqp_iter >= 95:
            self.get_logger().warn(
                f't={t_ref:.2f}s | KKT residual abnormal! res_stat={res_stat:.3e} '
                f'res_eq={res_eq:.3e} res_ineq={res_ineq:.3e} '
                f'res_comp={res_comp:.3e} | sqp_iter={sqp_iter} | '
                f'peak res_stat={self.max_res_stat:.3e}')

        if status == 0:
            self.solve_fail_count = 0
            u_opt = self.solver.get(0, 'u')
            # 角速度指令取下一预测节点 X_sol[10:13, 1](不是 u_opt 的力矩)。
            omega_cmd = self.solver.get(1, 'x')[10:13]
            self._q_pred1 = np.array(self.solver.get(1, 'x')[6:10], dtype=float)
            X_sol = np.array(
                [self.solver.get(i, 'x') for i in range(p.N + 1)]).T
            U_sol = np.array(
                [self.solver.get(i, 'u') for i in range(p.N)]).T
            self._set_nmpc_traj_msg(X_sol)
            # 存整条未 ramp 的角速度/推力轨迹给高频发布插值(copy 防被 shift 改到)。
            self._traj_omega = X_sol[10:13, :].copy()
            self._traj_thrust = U_sol[0, :].copy()

            # warm-start 向前滚动一步并复制末端
            for i in range(p.N):
                self.solver.set(i, 'x', X_sol[:, min(i + 1, p.N)])
                self.solver.set(i, 'u', U_sol[:, min(i + 1, p.N - 1)])
            self.solver.set(p.N, 'x', X_sol[:, p.N])

            self.last_u_opt = u_opt
            self.last_omega_cmd = omega_cmd
        else:
            # 没有可信轨迹:高频发布退化为零阶保持
            self._traj_omega = None
            self._traj_thrust = None
            self.solve_fail_count += 1
            self.get_logger().warn(
                f'acados solve failed [status={status}, '
                f'{self.solve_fail_count} consecutive] | '
                f'x_cur pos={x_cur[0:3]} vel={x_cur[3:6]} '
                f'|q|={np.linalg.norm(x_cur[6:10]):.4f}')

            if self.solve_fail_count > self.max_consecutive_fail:
                # 连续失败太久:求解器内部状态不可信,拉回安全悬停猜测
                self.get_logger().warn(
                    f'Exceeded {self.max_consecutive_fail} consecutive failures, '
                    f'resetting warm-start to safe hover state')
                self._seed_initial_guess(x_cur, Xref_win)
                u_opt = p.u_hover.copy()
                omega_cmd = np.zeros(3)
            elif self.solve_fail_count == 1:
                # 只失败这一帧:保留迭代值当下一次起点,指令延用上一次成功结果
                u_opt = self.last_u_opt.copy()
                omega_cmd = self.last_omega_cmd.copy()
            else:
                # 连续失败第 2 次起:推力延用,角速度必须归零(否则带着同一角速度持续旋转倾覆)
                u_opt = self.last_u_opt.copy()
                omega_cmd = np.zeros(3)

        self._traj_stamp = t_start_ros
        solve_time = (time.time() - t_start) * 1000
        return u_opt, omega_cmd, solve_time

    def _fd_reset(self):
        """清空诊断窗口累加器(每打印一行后调一次)。"""
        self._fd = {
            'tilt': 0.0,                    # 最大倾角(deg)
            'w': np.zeros(3),               # 机体角速度三轴绝对值峰值(rad/s)
            'u_frac': np.zeros(4),          # |u|/约束 的峰值 [T, tau_x, tau_y, tau_z]
            'u_sat': np.zeros(4),           # 各通道 >=98% 约束的帧数
            'om_lo': np.inf, 'om_hi': 0.0,  # omega_scale 窗口内 min/max
            'n': 0,
        }
        self._fd_mot_peak = 0.0
        self._fd_mot_sat = 0
        self._fd_mot_n = 0

    def _flight_diag_accum(self, u_opt):
        """逐帧累积飞行安全诊断量(只读)。饱和按约束值算:roll/pitch 上界 p.tau_max,
        yaw 是小得多的 p.tau_psi;推力上界 p.Tmax 是执行器真实上限。"""
        if self.x_cur is None:
            return
        d = self._fd
        q = self.x_cur[6:10]
        c = float(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))
        d['tilt'] = max(d['tilt'], math.degrees(math.acos(c)))
        d['w'] = np.maximum(d['w'], np.abs(self.x_cur[10:13]))
        u = np.abs(np.asarray(u_opt, dtype=float))
        lim = np.array([p.Tmax, p.tau_max, p.tau_max, p.tau_psi])
        frac = u / lim
        d['u_frac'] = np.maximum(d['u_frac'], frac)
        d['u_sat'] += (frac >= 0.98).astype(float)
        d['om_lo'] = min(d['om_lo'], self.omega_scale)
        d['om_hi'] = max(d['om_hi'], self.omega_scale)
        d['n'] += 1

    def _flight_diag_log(self, nmpc_time):
        """打一条 [flight-diag] 并清窗口。字段固定、单行,给离线聚合器解析。"""
        d = self._fd
        if d['n'] == 0:
            return
        n = float(d['n'])
        sat_pct = 100.0 * d['u_sat'] / n
        om_lo = 1.0 if not np.isfinite(d['om_lo']) else d['om_lo']
        mot_pct = (100.0 * self._fd_mot_sat / self._fd_mot_n
                   if self._fd_mot_n else float('nan'))
        tgt = self._payload_target
        hs = (f"health={int(bool(tgt.healthy))} "
              f"age={tgt.solution_age_sec:.3f}s "
              f"conf={self.no_payload_confidence:.3f}")
        self.get_logger().info(
            f'[flight-diag] t={nmpc_time:.1f}s | '
            f'om_scale={self.omega_scale:.3f}(lo {om_lo:.3f} hi {d["om_hi"]:.3f}) | '
            f'tilt_max={d["tilt"]:.2f}deg | '
            f'w_max=[r{d["w"][0]:.3f} p{d["w"][1]:.3f} y{d["w"][2]:.3f}]rad/s | '
            f'u_frac=[T{d["u_frac"][0]:.3f} r{d["u_frac"][1]:.3f} '
            f'p{d["u_frac"][2]:.3f} y{d["u_frac"][3]:.3f}] | '
            f'u_sat=[T{sat_pct[0]:.1f} r{sat_pct[1]:.1f} p{sat_pct[2]:.1f} '
            f'y{sat_pct[3]:.1f}]% | '
            f'mot_peak={self._fd_mot_peak:.3f} mot_sat={mot_pct:.1f}% | '
            f'{hs} | n={d["n"]}')
        self._fd_reset()

    def _update_omega_scale(self):
        """更新 ω_cmd 缩放比例 (Jxx+dJ_est)/Jxx(每次发布,滞回 + 一阶低通 + 不对称限速)。
        只缩放 roll/pitch:dJ 只进 Jxx/Jyy。"""
        dJ_eff = max(0.0, float(self.dJ_est))
        desired = float(np.clip(
            (p.Jxx + dJ_eff) / p.Jxx, 1.0, self.omega_scale_cap))
        # Schmitt-like dead band on the requested ratio.
        if desired <= 1.0 + 0.5 * self.omega_scale_hysteresis:
            # Hysteresis must not strand the empty-airframe steady state above one.
            self._omega_scale_target = 1.0
        elif desired >= self.omega_scale_cap - 0.5 * self.omega_scale_hysteresis:
            self._omega_scale_target = self.omega_scale_cap
        elif abs(desired - self._omega_scale_target) >= self.omega_scale_hysteresis:
            self._omega_scale_target = desired

        t = self.get_clock().now().nanoseconds * 1e-9   # ROS clock,不用墙钟
        if self._omega_scale_stamp is None:
            self._omega_scale_stamp = t
            return
        dt = t - self._omega_scale_stamp
        self._omega_scale_stamp = t
        if not (0.0 < dt < 0.5):            # 时基跳变帧:不推进
            return
        alpha = 1.0 - np.exp(-dt / max(self.omega_scale_tau, 1e-3))
        filtered = self.omega_scale + alpha * (
            self._omega_scale_target - self.omega_scale)
        rate = (self.omega_scale_rise_rate if filtered >= self.omega_scale
                else self.omega_scale_recover_rate)
        self.omega_scale = float(np.clip(
            slew(self.omega_scale, filtered, rate, dt),
            1.0, self.omega_scale_cap))

    def publish_attitude(self, u_opt, omega_cmd):
        T = u_opt[0]
        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 牛顿推力 T 反解成归一化油门: ω_req = sqrt(T/THRUST_K), norm = (ω_req-OMEGA_MIN)/OMEGA_SPAN
        if T > 0.0:
            omega_req = float(np.sqrt(T / self.THRUST_K))
            norm = (omega_req - self.OMEGA_MIN) / self.OMEGA_SPAN
        else:
            norm = 0.0
        # 边界与 p.Tmin/p.Tmax 同源;正常情况下只是兜底,真夹到了记一笔。
        norm_raw = norm
        norm = float(np.clip(norm, ACT_THROTTLE_MIN, ACT_THROTTLE_MAX))
        if abs(norm_raw - norm) > 1e-9:
            self._thr_clip_n = getattr(self, '_thr_clip_n', 0) + 1
            if self._thr_clip_n % 20 == 1:
                self.get_logger().warn(
                    f'[thr-clip] NMPC 求得 T={T:.2f}N → norm={norm_raw:.3f},'
                    f'被夹到 {norm:.3f}(=T {self.THRUST_K * (self.OMEGA_MIN + self.OMEGA_SPAN * norm) ** 2:.2f}N)。'
                    f'箱式约束是 [{p.Tmin:.2f},{p.Tmax:.2f}]N,本轮累计 {self._thr_clip_n} 次')
        msg.thrust = norm
        wmax = self.omega_cmd_max
        w_cmd = np.asarray(omega_cmd, dtype=float).copy()
        self._update_omega_scale()
        if self.omega_scale > 1.0 + 1e-6:
            # ω_cmd' = ω + s·(ω_cmd − ω),只作用 roll/pitch
            w_now = (self.x_cur[10:13] if self.x_cur is not None
                     else np.zeros(3))
            scale_eff = headroom_limited_scale(
                w_now[0:2], w_cmd[0:2], self.omega_scale,
                self.omega_scale_headroom_fraction * wmax)
            w_cmd[0:2] = (w_now[0:2]
                          + scale_eff * (w_cmd[0:2] - w_now[0:2]))
            if (scale_eff < self.omega_scale - 1e-6
                    and not self._omega_scale_clip_warned):
                self._omega_scale_clip_warned = True
                self.get_logger().warn(
                    f'[omega_scale] body-rate headroom limited scale '
                    f'{self.omega_scale:.2f}->{scale_eff:.2f} '
                    f'(guard ±{self.omega_scale_headroom_fraction*wmax:.2f}, '
                    f'hard ±{wmax:.2f}rad/s)')
        # [wcmd] 诊断:50Hz 下每 10 帧一行
        self._wlog_count = getattr(self, '_wlog_count', 0) + 1
        if self._wlog_count % 10 == 0:
            _wn = (self.x_cur[10:13] if self.x_cur is not None else np.zeros(3))
            _raw = np.asarray(omega_cmd, dtype=float)
            self.get_logger().info(
                f'[wcmd] raw=[{_raw[0]:+.3f},{_raw[1]:+.3f},{_raw[2]:+.3f}] '
                f'w=[{_wn[0]:+.3f},{_wn[1]:+.3f},{_wn[2]:+.3f}] '
                f'err=[{_raw[0]-_wn[0]:+.3f},{_raw[1]-_wn[1]:+.3f}] '
                f'scaled=[{w_cmd[0]:+.3f},{w_cmd[1]:+.3f}] '
                f's={self.omega_scale:.2f}')
        msg.body_rate.x = float(np.clip(w_cmd[0], -wmax, wmax))
        msg.body_rate.y = float(np.clip(w_cmd[1], -wmax, wmax))
        msg.body_rate.z = float(np.clip(w_cmd[2], -wmax, wmax))
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        self.att_pub.publish(msg)

    def _publish_from_traj(self):
        """高频发布定时器:沿最近一次求解的预测轨迹插值前推,发 body_rate + thrust。
        只在 NMPC 接管后生效。单线程 executor 下与 timer_cb 串行,无需加锁。"""
        if not self.nmpc_started or self.nmpc_start_time is None:
            return
        nmpc_time = (self.get_clock().now()
                     - self.nmpc_start_time).nanoseconds / 1e9

        if self._traj_omega is not None:
            # 角速度看前一步(lookahead=dt),推力取当前步;τ clamp 到网格末端。
            tau = (self.get_clock().now() - self._traj_stamp).nanoseconds / 1e9
            t_om = min(max(p.dt + tau, 0.0), self._omega_grid[-1])
            t_th = min(max(tau, 0.0), self._thrust_grid[-1])
            omega_cmd = np.array([
                np.interp(t_om, self._omega_grid, self._traj_omega[k])
                for k in range(3)])
            T = float(np.interp(t_th, self._thrust_grid, self._traj_thrust))
        else:
            # 求解失败(或还没有第一条轨迹):零阶保持最近一次成功的指令
            omega_cmd = self.last_omega_cmd.copy()
            T = float(self.last_u_opt[0])

        # body_rate 接管斜坡(smoothstep)
        if nmpc_time < self.bodyrate_ramp_time:
            s = nmpc_time / self.bodyrate_ramp_time
            omega_cmd = omega_cmd * (10*s**3 - 15*s**4 + 6*s**5)

        self.publish_attitude(np.array([T, 0.0, 0.0, 0.0]), omega_cmd)

    # ------------------------------------------------------------ 任务阶段
    def _grip_ref(self, t=0.0, **kw):
        """固定悬停在 (grip_x, grip_y, self.z_hover),姿态水平、速度/角速度为零。"""
        return np.concatenate([
            np.array([self.grip_x, self.grip_y, self.z_hover]),
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)])

    def _grip_ref_window(self, t_start, N, dt, z_hover=None):
        return np.tile(self._grip_ref().reshape(-1, 1), (1, N + 1))

    def _grip_dyn_ref(self, t=0.0, **kw):
        xr = build_reference_figure8(
            t - self.grip_dyn_t0, r=self.grip_dyn_r, w=self.grip_dyn_w,
            z_hover=self.grip_z_high, dz=self.grip_dyn_dz,
            ramp_time=self.grip_dyn_ramp, yaw_ramp=True)
        xr[0] += self.grip_dyn_cx
        xr[1] += self.grip_dyn_cy
        return xr

    def _grip_dyn_ref_window(self, t_start, N, dt, z_hover=None):
        xref = build_reference_window_figure8(
            t_start - self.grip_dyn_t0, N, dt, r=self.grip_dyn_r,
            w=self.grip_dyn_w, z_hover=self.grip_z_high, dz=self.grip_dyn_dz,
            ramp_time=self.grip_dyn_ramp, yaw_ramp=True)
        xref[0, :] += self.grip_dyn_cx
        xref[1, :] += self.grip_dyn_cy
        return xref

    def _grip_dynamic_phase(self, nmpc_time):
        """lift 完成 + grip_dyn_settle_sec 后从悬停切到 8 字(偏移到切换瞬间的 xy)。只切一次。"""
        if (self._grip_dyn_latched or not self.grip_lift_started
                or self.attach_time is None or self.x_cur is None):
            return
        lift_done_t = (self.attach_time + self.grip_lift_after_sec
                       + self.grip_lift_dur)
        if nmpc_time < lift_done_t + self.grip_dyn_settle_sec:
            return
        self.grip_dynamic_active = True
        self._grip_dyn_latched = True
        self.grip_dyn_t0 = nmpc_time
        self.grip_dyn_cx = float(self.x_cur[0])
        self.grip_dyn_cy = float(self.x_cur[1])
        self.hover_test_mode = False   # 放开 t_ref,让 8 字随时间推进
        self.ref_fn = self._grip_dyn_ref
        self.ref_window_fn = self._grip_dyn_ref_window
        # RViz 参考线按新 ref_fn 重建
        self.ref_path_msg = self._build_ref_path_msg(
            w=self.grip_dyn_w, z_hover=self.grip_z_high,
            ramp_time=self.grip_dyn_ramp, t_offset=self.grip_dyn_t0)
        self.get_logger().info(
            f't={nmpc_time:.1f}s | DYNAMIC: switch to figure8 (r={self.grip_dyn_r} '
            f'w={self.grip_dyn_w}) centered at [{self.grip_dyn_cx:.2f},'
            f'{self.grip_dyn_cy:.2f}] z={self.grip_z_high}, tracking with payload')

    def _grip_mass_step(self, nmpc_time):
        """attach 由本节点自己的 enable 指令定时(不订阅夹爪的 attach 通知)。
        m/s 完全来自 MHE;只给 J 一个临时的抬升前安全地板。"""
        if self.grip_mass_stepped:
            return
        if not self.grip_descend_done or self.grip_descend_done_time is None:
            return
        self.grip_mass_stepped = True
        self.attach_time = self.grip_descend_done_time
        self._reset_acc_loss_watchdog()
        self._start_attach_j_bootstrap()
        self.get_logger().info(
            f't={nmpc_time:.1f}s | ATTACH command issued; m/s remain on '
            'continuous MHE estimates, J safety bootstrap active '
            '(no attach notification subscription)')

    def _descend_phase(self, nmpc_time):
        """NMPC 在安全高度接管后先稳定 grip_settle_sec,再平滑下降到 grip_z_low;
        斜坡完成后等真实高度/速度收敛、对准就绪才发 enable 触发 attach。"""
        if self.grip_descend_done:
            return
        if nmpc_time < self.grip_settle_sec:
            return
        s = min(max((nmpc_time - self.grip_settle_sec) /
                    self.grip_descend_dur, 0.0), 1.0)
        smooth = 10*s**3 - 15*s**4 + 6*s**5
        self.z_hover = self.grip_approach_z + \
            (self.grip_z_low - self.grip_approach_z) * smooth
        if not self.grip_descend_started:
            self.grip_descend_started = True
            self.get_logger().info(
                f't={nmpc_time:.1f}s | DESCEND: settle done, lowering '
                f'{self.grip_approach_z}->{self.grip_z_low}m, will enable '
                f'gripper at bottom')
        z_err = float(abs(self.x_cur[2] - self.grip_z_low))
        v_norm = float(np.linalg.norm(self.x_cur[3:6]))
        if (s >= 1.0 and not self.grip_enable_sent
                and z_err < self.grip_descend_z_tol
                and v_norm < self.grip_descend_v_tol
                and self._align_ready(nmpc_time)):
            self.grip_descend_done = True
            self.grip_descend_done_time = nmpc_time
            self.grip_enable_sent = True
            self.enable_pub.publish(Bool(data=True))
            self.get_logger().info(
                f't={nmpc_time:.1f}s | DESCEND done at '
                f'z={self.x_cur[2]:.3f}m (target {self.grip_z_low:.3f}m, '
                f'z_err={z_err:.3f}m, v={v_norm:.2f}m/s), gripper ENABLED '
                f'(attach will trigger now)')

    def target_rel_cb(self, msg):
        if len(msg.data) >= 2:
            self._target_rel = np.asarray(msg.data[:2], dtype=float)
            self._target_rel_wall = time.monotonic()

    def _target_rel_fresh(self):
        return (self._target_rel is not None
                and time.monotonic() - self._target_rel_wall < 0.3)

    def _update_grip_align(self, nmpc_time):
        """吸附使能前,把 (grip_x, grip_y) 低通跟到"本地系 box + offset"。
        只在水平速度 < 0.3m/s 时更新;使能后冻结。"""
        if (self.grip_enable_sent or self.x_cur is None
                or not self._target_rel_fresh()):
            return
        if float(np.linalg.norm(self.x_cur[3:5])) > 0.3:
            return
        target = self.x_cur[0:2] + self._target_rel + self.grip_align_offset
        if self._align_xy is None:
            self._align_xy = target.copy()
        else:
            alpha = 1.0 - math.exp(-p.dt / self.grip_align_tau)
            self._align_xy += alpha * (target - self._align_xy)
        self.grip_x, self.grip_y = (float(v) for v in self._align_xy)
        if nmpc_time - self._align_log_t >= 1.0:
            self._align_log_t = nmpc_time
            self.get_logger().info(
                f't={nmpc_time:.1f}s | [align] target=({self.grip_x:+.3f},'
                f'{self.grip_y:+.3f}) err_xy={self._align_err():.3f}m '
                f'rel=({self._target_rel[0]:+.3f},{self._target_rel[1]:+.3f})')

    def _align_err(self):
        """当前真实水平对准误差 = |(机体 - box) - offset| = |-rel - offset|。"""
        return float(np.linalg.norm(-self._target_rel - self.grip_align_offset))

    def _align_ready(self, nmpc_time):
        """使能门控:对准误差进容差;等满 wait_max 仍不行就放行并告警。"""
        if self._align_wait_start is None:
            self._align_wait_start = nmpc_time
        waited = nmpc_time - self._align_wait_start
        if self._target_rel_fresh() and self._align_err() < self.grip_align_xy_tol:
            self.get_logger().info(
                f't={nmpc_time:.1f}s | [align] ready: err_xy={self._align_err():.3f}m '
                f'after {waited:.1f}s')
            return True
        if waited >= self.grip_align_wait_max:
            err = (f'{self._align_err():.3f}m' if self._target_rel_fresh()
                   else 'no fresh target_rel')
            self.get_logger().warn(
                f't={nmpc_time:.1f}s | [align] not aligned after {waited:.1f}s '
                f'(err_xy={err}), enabling anyway')
            return True
        return False

    def _lift_phase(self, nmpc_time):
        """attach 满 grip_lift_after_sec 后把 z_hover 从 grip_z_low 平滑抬到 grip_z_high。"""
        if (self.attach_time is None
                or nmpc_time < self.attach_time + self.grip_lift_after_sec):
            return
        t_eff = nmpc_time - self.attach_time - self.grip_lift_after_sec
        s = min(max(t_eff / self.grip_lift_dur, 0.0), 1.0)
        smooth = 10*s**3 - 15*s**4 + 6*s**5
        self.z_hover = self.grip_z_low + \
            (self.grip_z_high - self.grip_z_low) * smooth
        if not self.grip_lift_started:
            self.grip_lift_started = True
            self.get_logger().info(
                f't={nmpc_time:.1f}s | LIFT: raising hover '
                f'{self.grip_z_low}->{self.grip_z_high}m to lift payload')

    def _grip_drop_phase(self, nmpc_time):
        """lift 完成后至少 grip_drop_after_sec,在 8 字最左端(a=3π/2)发释放指令
        (/gripper/enable=false)。模型不随指令改变:完成与 L1 复位只在持续确认之后。"""
        if (self.grip_drop_done or self.grip_drop_pending
                or self.grip_drop_after_sec <= 0.0 or self.attach_time is None):
            return
        t_drop = (self.attach_time + self.grip_lift_after_sec
                  + self.grip_lift_dur + self.grip_drop_after_sec)
        if nmpc_time < t_drop:
            return
        # 尖点窗口宽度跟着 w 走(至少 3 帧),否则高速时会整圈漏过。
        if self.grip_dynamic_active:
            tc = nmpc_time - self.grip_dyn_t0 - 2.0  # 2.0 = figure8 hover_time
            if tc <= 0.0:
                return
            a_mod = (self.grip_dyn_w * tc) % (2.0 * np.pi)
            tip_win = max(0.12, 3.0 * self.grip_dyn_w * p.dt)
            if not (1.5 * np.pi <= a_mod < 1.5 * np.pi + tip_win):
                return
        if self.grip_uncommanded_loss:
            # 故障注入:走 /gripper/release 物理脱离,控制器与 MHE 都收不到任何释放指令。
            self.grip_drop_done = True
            self.grip_release_pub.publish(Bool(data=True))
            self.get_logger().warn(
                f't={nmpc_time:.1f}s | UNCOMMANDED LOSS injected: physical release '
                'via /gripper/release; no release command issued')
            self.get_logger().info(evt_line('inject_loss', t=nmpc_time))
            return
        self.enable_pub.publish(Bool(data=False))  # 拉低 → proximity 释放 box
        self.get_logger().info(evt_line('cmd', t=nmpc_time, continuous=True))
        self.grip_drop_pending = True
        self._drop_cmd_wall = self.get_clock().now().nanoseconds * 1e-9
        # 释放指令是明确边沿,无条件重新武装空载检测(否则 latch 早被消耗,DROP 永不完成)。
        self._no_payload_latched = False
        self._no_payload_conf_frames = 0
        self.get_logger().info(
            f't={nmpc_time:.1f}s | DROP command issued; waiting for MHE '
            f'no-payload confidence >= {self.no_payload_conf_threshold:.2f} '
            f'for {self.no_payload_conf_hold_frames * p.dt:.2f}s')

    # ------------------------------------------------------------ 主循环
    def timer_cb(self):
        if self.counter < 100:
            self.pub_hover_pos()
            self.counter += 1
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now()

        t_elapsed = (self.get_clock().now() -
                     self.start_time).nanoseconds / 1e9

        if not self.state.connected:
            self.pub_hover_pos()
            return

        if t_elapsed < self.ekf_wait_sec:
            self.pub_hover_pos()
            self.counter += 1
            if self.counter % 100 == 0:
                self.get_logger().info(
                    f'Waiting for EKF2 convergence... '
                    f'{t_elapsed:.0f}/{self.ekf_wait_sec:.0f} s')
            return

        now = time.time()

        if self.state.mode != 'OFFBOARD':
            self.pub_hover_pos()
            if (self.mode_client.service_is_ready()
                    and now - self.last_mode_req_time > 1.0):
                req = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                self.mode_client.call_async(req)
                self.last_mode_req_time = now
                self.get_logger().info('Switching to OFFBOARD mode')
            return

        if not self.state.armed:
            self.armed_and_flying = False
            self.nmpc_started = False
            self.nmpc_start_time = None
            self.pub_hover_pos()
            if (self.arming_client.service_is_ready()
                    and now - self.last_arm_req_time > 1.0):
                req = CommandBool.Request()
                req.value = True
                self.arming_client.call_async(req)
                self.last_arm_req_time = now
                self.get_logger().info('Sending arm command')
            return

        if not self.armed_and_flying:
            self.armed_and_flying = True
            self.get_logger().info('Armed, preparing to move to trajectory start...')

        if self.x_cur is None:
            self.pub_hover_pos()
            return

        start_ref = self.ref_fn(0.0)
        err_xy = float(np.linalg.norm(self.x_cur[0:2] - start_ref[0:2]))
        err_z  = float(abs(self.x_cur[2] - start_ref[2]))
        v_norm = float(np.linalg.norm(self.x_cur[3:6]))
        # 接近第 0 段:先在起飞点原地垂直爬到安全高度(box 是实体障碍,不能低空平移)。
        if not self.nmpc_started and not self.grip_climbed:
            climb_ref = start_ref.copy()
            climb_ref[0] = 0.0
            climb_ref[1] = 0.0
            climb_ref[2] = self.grip_approach_z
            self.pub_position_ref(climb_ref)
            self.counter += 1
            if self.counter % 50 == 0:
                self.get_logger().info(
                    f'Gripper approach (climb): rising at spawn to '
                    f'z={self.grip_approach_z:.2f}m | z={self.x_cur[2]:.2f}m '
                    f'v={v_norm:.2f}m/s')
            if (abs(self.x_cur[2] - self.grip_approach_z) < 0.20
                    and v_norm < 0.30):
                self.grip_climbed = True
                self.get_logger().info(
                    'Gripper approach: climbed to safe altitude at spawn, '
                    'now translating over box')
            return

        # 接近第 1 段:在安全高度平移到 box 正上方并悬停稳。
        if not self.nmpc_started and not self.grip_high_aligned:
            high_ref = start_ref.copy()
            high_ref[2] = self.grip_approach_z
            self.pub_position_ref(high_ref)
            hi_xy = float(np.linalg.norm(self.x_cur[0:2] - high_ref[0:2]))
            hi_z = float(abs(self.x_cur[2] - high_ref[2]))
            self.counter += 1
            if self.counter % 50 == 0:
                self.get_logger().info(
                    f'Gripper approach (high): aligning over box at '
                    f'z={self.grip_approach_z:.2f}m | xy={hi_xy:.3f}m '
                    f'z_err={hi_z:.3f}m v={v_norm:.2f}m/s')
            if hi_xy < 0.10 and hi_z < 0.20 and v_norm < 0.25:
                self.grip_high_aligned = True
                self.get_logger().info(
                    'Gripper approach: aligned above box, NMPC will take '
                    f'over here (z={self.grip_approach_z:.2f}m), settle '
                    f'{self.grip_settle_sec:.1f}s then descend to '
                    f'z={self.grip_z_low:.2f}m to trigger attach')
            return

        if not self.nmpc_started:
            self.pub_position_ref(start_ref)
            self.counter += 1
            if self.counter % 50 == 0:
                self.get_logger().info(
                    f'Moving to trajectory start... xy={err_xy:.3f}m '
                    f'z={err_z:.3f}m v={v_norm:.2f}m/s')
            if (err_xy > self.start_xy_threshold
                    or err_z > self.start_z_threshold
                    or v_norm > 0.25):
                return

            self.nmpc_started = True
            self.nmpc_start_time = self.get_clock().now()
            self._seed_initial_guess(
                start_ref, np.tile(start_ref.reshape(-1, 1), (1, p.N + 1)))
            self.get_logger().info('Reached trajectory start, NMPC tracking begins!')

        nmpc_time = (self.get_clock().now() -
                     self.nmpc_start_time).nanoseconds / 1e9
        self._update_grip_align(nmpc_time)
        self._descend_phase(nmpc_time)
        self._grip_mass_step(nmpc_time)
        self._lift_phase(nmpc_time)
        self._grip_dynamic_phase(nmpc_time)
        self._grip_drop_phase(nmpc_time)
        t_ref = 0.0 if self.hover_test_mode else nmpc_time

        u_opt, omega_cmd, solve_time = self.solve_nmpc(self.x_cur, t_ref)
        # body_rate/thrust 由高频 _publish_from_traj 发布;u_opt 以求解频率发布。
        self.u_opt_pub.publish(Float64MultiArray(data=[float(v) for v in u_opt]))

        xref_now = self.ref_fn(t_ref)
        _e = self.x_cur[0:3] - xref_now[0:3]
        pos_err = np.linalg.norm(_e)
        self._flight_diag_accum(u_opt)
        self.tracking_err_pub.publish(Float64(data=float(pos_err)))

        # [model-used] 10Hz:本周期实际装进 model.p 的载荷参数与生命周期标志,
        # 供评分脚本算 t_empty 与模型—物理不一致时长。纯只读。
        if self.counter % max(1, round(0.1 / p.dt)) == 0:
            self.get_logger().info(
                f'[model-used] t={nmpc_time:.2f} pos_err={pos_err:.3f} '
                f'm={self.m_est:.4f} g0={self.dJ_est:.5f} g1={0.0:.5f} g2={0.0:.5f} '
                f'mode=cont conf={self.no_payload_confidence:.3f} '
                f'fresh={int(bool(self._payload_estimate_health_prev))} '
                f'pending={int(bool(self.grip_drop_pending))} '
                f'latched={int(bool(self._no_payload_latched))} '
                f'dropped={int(bool(self.grip_dropped))} '
                f'unresolved={int(bool(self.payload_unresolved))} '
                f'T={u_opt[0]:.2f} tr={u_opt[1]:.3f} tp={u_opt[2]:.3f} '
                f'e=[{_e[0]:+.3f},{_e[1]:+.3f},{_e[2]:+.3f}] '
                f'att_err={self._att_err_deg():.2f} '
                f'd=[{self.d_lumped[0]:+.3f},{self.d_lumped[1]:+.3f},'
                f'{self.d_lumped[2]:+.3f}]')

        # [attach-window] attach 后逐帧记录(所有 A/B 分析的数据源)
        if self.attach_time is not None:
            _dt_ev = nmpc_time - self.attach_time
            if 0.0 <= _dt_ev < self.attach_window_sec:
                self.get_logger().info(
                    f'[attach-window] t={nmpc_time:.2f}s pos_err={pos_err:.3f}m '
                    f'T={u_opt[0]:.2f}N z={self.x_cur[2]:.3f} '
                    f'm_est={self.m_est:.3f}')

        self.counter += 1
        # 约每 5 秒一条摘要日志
        if self.counter % max(1, round(5.0 / p.dt)) == 0:
            self.get_logger().info(
                f't={nmpc_time:.1f}s | pos_err={pos_err:.3f}m | '
                f'T={u_opt[0]:.2f}N | tau=[r{u_opt[1]:.3f} p{u_opt[2]:.3f} '
                f'y{u_opt[3]:.3f}]Nm | '
                f'res_stat={self.last_res_stat:.3e}(peak {self.max_res_stat:.3e}) '
                f'sqp_iter={self.last_sqp_iter} | solve={solve_time:.1f}ms')
            self._flight_diag_log(nmpc_time)


def main():
    rclpy.init()
    node = AcadosNMPCNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
