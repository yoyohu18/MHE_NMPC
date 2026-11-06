#!/usr/bin/env python3
# 物理/安全常数集中在本包 common.py；Q/R/P/N/dt 是 NMPC 自己的配置。


import numpy as np
from .common import AirframeParams

_base = AirframeParams()

# ===== 执行器真实推力包线(2026-09-10)=====
# 原来 Tmin/Tmax 直接取 _base 的 0.5N / 2·m·g=40.50N,那是**按飞机重量拍的**,
# 跟执行器实际能发出多少推力没有关系。真实链路(与 acados_nmpc_node.publish_attitude
# 里那条物理映射同源,见该处注释)是:
#   norm(归一化油门,发给 PX4 的 AttitudeTarget.thrust)
#     → ω = OMEGA_MIN + OMEGA_SPAN·norm      (GZMixingInterfaceESC, SIM_GZ_EC_MIN/MAX)
#     → T = THRUST_K·ω²                       (gz MulticopterMotorModel, 4 电机合计)
# 而 publish_attitude 又把 norm 夹在 [0.05, 0.95](上界留 5% 给姿态环做差动,
# 否则四个电机全打满时力矩权限归零 —— 这个 0.95 是对的,不该动)。于是:
#
#   NMPC 以为的上限   2·m_B·g   = 40.50 N
#   实际能发的上限   norm=0.95 = 31.35 N      ← 差 29%
#   硬件极限         norm=1.00 = 34.19 N
#   NMPC 以为的下限   原值 0.50 N
#   实际能发的下限   norm=0.05 =  1.27 N      ← 差 2.5×
#
# 后果不是"约束保守了一点",是**规划与执行脱钩**:带 0.3kg 载荷时悬停 23.2N,
# NMPC 以为还有 17.3N 余量(0.75g),真实只有 8.2N(0.35g)—— 它规划的恢复加速度
# 可以是实际的 2.1 倍。规划跟不上 → 误差扩大 → 下一拍要更多推力 → 仍被 clip,
# 正是"闭环失稳而非求解器发散"所需的正反馈条件。下界同理:那些"推力塌到下界
# 0.50N"的发散案例里,0.50N 这个数飞机根本发不出来。
#
# clip 还是**静默**的:飞行诊断的 u_frac[T] 拿 40.50 当分母,真实饱和点落在
# u_frac=0.774,所以日志里 u_sat[T] 恒为 0.0% —— 推力从来没"显示"饱和过。
# 对齐之后 u_frac[T]=1.0 才真正等于"推力打满",那条诊断随之变得可信。
#
# ⚠️ 改这两个数会让 acados 重新生成+编译 solver(ubu/lbu 变了,
#    is_code_reuse_possible 会判定不可复用),首次启动多花 10-30 秒,自动发生。
THRUST_K     = 4.0 * 8.54858e-06  # T=THRUST_K·ω² [N/(rad/s)²],4 电机合计
OMEGA_MIN    = 150.0              # SIM_GZ_EC_MIN
OMEGA_SPAN   = 850.0              # SIM_GZ_EC_MAX(1000) − SIM_GZ_EC_MIN(150)
THROTTLE_MIN = 0.05               # publish_attitude 的 clip 下界
THROTTLE_MAX = 0.95               # publish_attitude 的 clip 上界(留给姿态环差动)


def thrust_at_throttle(norm: float) -> float:
    """归一化油门 → 总推力(N)。publish_attitude 的正向映射,单一真相源。"""
    w = OMEGA_MIN + OMEGA_SPAN * float(norm)
    return THRUST_K * w * w


T_ACT_MIN = thrust_at_throttle(THROTTLE_MIN)   # 1.27 N
T_ACT_MAX = thrust_at_throttle(THROTTLE_MAX)   # 31.35 N

class AcadosParams:
    # --- 物理常数统一定义在本包 common.AirframeParams。---
    m   = _base.m    # 机体质量(kg)。出现在推力方程 vel_dot=(1/m)(R@[0,0,T]-kd*v)-g 里
                      # Mass (kg). Appears in vel_dot = (1/m)(R@[0,0,T] - kd*v) - g
    g   = _base.g    # 重力加速度(m/s^2),ENU 坐标系(Z 朝上),方程里是 -g_vec(沿 -Z)
                      # Gravitational acceleration (m/s^2), ENU frame (Z up); enters as -g_vec
    Jxx = _base.Jxx  # 绕机体 x 轴(roll)转动惯量(kg*m^2),用在角速度动力学 om_dot=(tau-omx(J*om))/J
                      # Moment of inertia about body x-axis (roll, kg*m^2), used in om_dot=(tau-omx(J*om))/J
    Jyy = _base.Jyy  # 绕机体 y 轴(pitch)转动惯量(kg*m^2)
                      # Moment of inertia about body y-axis (pitch, kg*m^2)
    Jzz = _base.Jzz  # 绕机体 z 轴(yaw)转动惯量(kg*m^2),比 Jxx/Jyy 大,yaw 方向惯性更强
                      # Moment of inertia about body z-axis (yaw, kg*m^2); larger than Jxx/Jyy
    kd  = _base.kd   # 线性空气阻力系数,vel_dot 里的 -kd*vel 项,模拟桨叶/机身阻力
                      # Linear drag coefficient, the -kd*vel term in vel_dot (models prop/airframe drag)
    # 推力箱式约束 = 执行器真实包线(见文件头 T_ACT_MIN/MAX 那段)。
    Tmin    = T_ACT_MIN
                            # 总推力下限(N),acados 输入约束 lbu[0]。
                            # Min total thrust (N) = actuator floor at 5% throttle
    Tmax    = T_ACT_MAX
                            # 总推力上限(N),acados 输入约束 ubu[0]。
                            # Max total thrust (N) = actuator ceiling at 95% throttle
    tau_max = _base.tau_max # roll/pitch 力矩约束上限(Nm),输入约束 ubu[1],ubu[2]
                              # Roll/pitch torque limit (Nm), input constraint ubu[1], ubu[2]
    tau_psi = _base.tau_psi # yaw 力矩约束上限(Nm),比 tau_max 小很多(0.2 vs 0.5)——
                             # 实测电机能输出的偏航力矩本来就比横滚/俯仰小
                             # Yaw torque limit (Nm), much smaller than tau_max (0.2 vs 0.5) —
                             # motors simply can't produce as much yaw torque as roll/pitch
    # --- 质量符号约定(2026-08-24 全仓统一)---
    # m_B = 空机质量(已知常数,就是上面的 m);m_P = 载荷质量(未知);m_T = m_B+m_P。
    # NMPC 里 model.p 的质量槛位装的是 **m_T**(来自 MHE 的 self.m_est);p.m 只在
    # "空机标定值/兜底初值/Tmax 基准"这三处出现,永远是 m_B。
    m_B = _base.m

    nq_err = 3   # 姿态误差维数:q_ref* ⊗ q 的向量部分
    nx = 13  # 状态维度: pos(3)+vel(3)+quat(4)+omega(3) = 13
              # State dimension: pos(3)+vel(3)+quat(4)+omega(3) = 13
    nu = 4   # 控制维度: 总推力 T(1) + 力矩 tau_x,tau_y,tau_z(3) = 4
              # Control dimension: total thrust T(1) + torques tau_x,tau_y,tau_z(3) = 4

    # --- acados 控制器自己的时域/代价权重。---
    # acados controller's own horizon/cost weights, tuned independently
    # These values are independent of the shared physical constants.
    # 20 Hz NMPC: dt 同时是离散步长和 ROS2 求解定时器周期。N 随 dt 从原来的
    # 10@0.1s 同步增至 20，保持 N*dt=1.0s 的预测时域不变。
    # 旧实验曾观察到 20 Hz 配置的 attach 瞬态和求解耗时变差，因此切换频率后
    # 必须重新做闭环验收；这里仍按当前要求启用 20 Hz，而不是缩短预测时域。
    # 20 Hz NMPC: dt is both the shooting interval and ROS2 solve-timer period.
    # N increases with the rate so the 1.0 s prediction horizon is preserved.
    N  = 20
    dt = 0.05

    # --- Bryson's rule 无量纲化: Q_ii = 1/(典型尺度)^2,R_jj 同理 ---
    # Bryson's rule non-dimensionalization: Q_ii = 1/(typical scale)^2, same for R_jj.
    # 诊断发现 res_stat 起点 2.16、收敛率卡在 ~0.9(干净环境复测下确认是真实
    # 问题,不是环境干扰:res_stat 峰值能飙到 6.26e5,彻底数值崩溃)。根因之一
    # 是 Q_pos=4 对 R_torque=0.1 的量纲失衡,Hessian 在不同方向曲率差几十倍。
    # 每个分量在各自"典型尺度"下对 cost 的贡献都归一化到 1,让 Gauss-Newton
    # 的 Hessian 各方向曲率均衡。下面这组 L_* 就是各状态/控制误差分量的
    # "容许误差量级"(单位跟对应物理量一致),取得越小代表对应方向罚得越重。
    # Diagnosis found res_stat starting at 2.16 with convergence rate stuck at
    # ~0.9 (confirmed real, not environment noise, via a clean-env retest —
    # res_stat could spike to 6.26e5, a full numerical blow-up). One root cause:
    # Q_pos=4 vs R_torque=0.1 is a unit mismatch, making the Hessian's curvature
    # differ by tens of times across directions. Normalizing each component's
    # cost contribution to 1 at its own "typical scale" balances the Gauss-Newton
    # Hessian's curvature across directions. The L_* values below are each
    # state/control error's "tolerable error magnitude" (same units as the
    # physical quantity) — smaller L means that direction is penalized harder.
    L_pos   = 0.1       # 位置误差容许量级(m):0.1m 的跟踪误差算"正常"
                          # Tolerable position error (m): 0.1m tracking error is "normal"
    L_vel   = 0.3       # 速度误差容许量级(m/s) = 轨迹特征速度 r*w(半径1m*角速度0.3rad/s)
                          # Tolerable velocity error (m/s) = trajectory's characteristic speed r*w
    L_att   = 0.05      # 姿态误差(四元数虚部)容许量级,约对应 5.7° 的姿态偏差
                          # Tolerable attitude error (quaternion vector part), ~5.7 degrees
    L_omega = 0.3       # 角速度误差容许量级(rad/s),跟圆形轨迹的恒定 yaw_rate 同量级
                          # Tolerable angular-rate error (rad/s), same order as the circle's constant yaw_rate
    L_T     = 2.0       # 推力误差容许量级(N),相对悬停推力(~20N)的合理波动范围
                          # Tolerable thrust error (N), a reasonable swing around hover thrust (~20N)
    L_tau_rp  = tau_max  # roll/pitch 力矩误差容许量级,直接取约束值本身(控制量天然不会超约束)
                           # Tolerable roll/pitch torque error — just use the constraint itself
                           # (the control input can never exceed it anyway)
    L_tau_yaw = tau_psi  # yaw 力矩误差容许量级,同理取 tau_psi
                           # Tolerable yaw torque error, likewise taken as tau_psi

    # cost_y_expr = [tracking_error_sym(x,xr)(12维); u - u_hover(4维)] -> 16维
    # cost_y_expr = [tracking_error_sym(x,xr) (12-dim); u - u_hover (4-dim)] -> 16-dim
    # Q 作用在 tracking_error_sym 的 12 维输出上,按 4 个 3x3 块对应:
    # Q acts on the 12-dim output of tracking_error_sym, as 4 blocks of 3x3:
    #   block 1 = 位置误差(ep)权重 = 1/L_pos^2  = 100
    #             position error (ep) weight = 1/L_pos^2 = 100
    #   block 2 = 速度误差(ev)权重 = 1/L_vel^2  ≈ 11.1
    #             velocity error (ev) weight = 1/L_vel^2 ≈ 11.1
    #   block 3 = 姿态误差(eq_vec,四元数虚部)权重 = 1/L_att^2 = 400
    #             attitude error (eq_vec, quaternion vector part) weight = 1/L_att^2 = 400
    #   block 4 = 角速度误差(eomega)权重 = 1/L_omega^2 ≈ 11.1
    #             angular-rate error (eomega) weight = 1/L_omega^2 ≈ 11.1
    Q = np.diag([1/L_pos**2]*3 + [1/L_vel**2]*3 + [1/L_att**2]*nq_err
                + [1/L_omega**2]*3)
    # R 作用在 u-u_hover 的 4 维残差上,对角线依次是:
    # R acts on the 4-dim residual u-u_hover; the diagonal entries are:
    #   [0] 推力误差权重    = 1/L_T**2      = 0.25
    #       thrust error weight = 1/L_T**2 = 0.25
    #   [1] roll 力矩误差权重 = 1/L_tau_rp**2  = 4.0
    #       roll torque error weight = 1/L_tau_rp**2 = 4.0
    #   [2] pitch 力矩误差权重 = 1/L_tau_rp**2 = 4.0(跟 roll 共用同一个典型尺度)
    #       pitch torque error weight = 1/L_tau_rp**2 = 4.0 (shares the same scale as roll)
    #   [3] yaw 力矩误差权重  = 1/L_tau_yaw**2 = 25.0(yaw 力矩约束更紧,所以罚得更重)
    #       yaw torque error weight = 1/L_tau_yaw**2 = 25.0 (tighter constraint -> heavier penalty)
    R = np.diag([1/L_T**2, 1/L_tau_rp**2, 1/L_tau_rp**2, 1/L_tau_yaw**2])

    # 终端代价(cost_y_expr_e,只有 12 维 tracking_error_sym,没有控制项)跟
    # stage 的 Q 保持相同的相对比例放大,逼着预测轨迹末端更精确地落在参考点上:
    #   位置 20 倍(2000)、速度 1 倍(11.1,不额外加重)、姿态 2 倍(800)、
    #   角速度 0.2 倍(2.22,末端不需要严格限制角速度,放松反而利于求解)
    # Terminal cost (cost_y_expr_e, just the 12-dim tracking_error_sym, no control
    # term) scales the stage Q by the same relative factors, forcing the predicted
    # trajectory's endpoint closer to the reference:
    #   position x20 (2000), velocity x1 (11.1, no extra weight), attitude x2 (800),
    #   angular rate x0.2 (2.22 — no need to tightly constrain terminal angular
    #   rate; relaxing it actually helps the solver)
    P = np.diag([20*(1/L_pos**2)]*3 + [1/L_vel**2]*3 + [2*(1/L_att**2)]*nq_err
                + [0.2*(1/L_omega**2)]*3)

    # 悬停时的参考控制量:推力=重力(T=m*g,刚好抵消重力),三个力矩都是 0
    # (悬停不需要任何转动)。cost 里 u-u_hover 算的就是相对这个基准的偏差,
    # 而不是相对绝对零的偏差——这样悬停时残差天然是 0,不会被无谓惩罚。
    # Reference control at hover: thrust = gravity (T=m*g, exactly cancels
    # gravity), all three torques are 0 (no rotation needed at hover). The cost
    # computes u-u_hover relative to this baseline, not relative to absolute
    # zero — so the residual is naturally 0 at hover and isn't penalized for no reason.
    u_hover = np.array([m * g, 0.0, 0.0, 0.0])


p = AcadosParams()
