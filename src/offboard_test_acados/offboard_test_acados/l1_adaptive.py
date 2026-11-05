#!/usr/bin/env python3
# C.1 L1-NMPC baseline 的 L1 自适应增广(2026-07-23,实施计划_C1_..._20260723.md §2/§4)。
#
# 定位:阶段 C 对照实验的"流派 B"——把变载荷当 **lumped 平动扰动**在线估计并补偿,
# **不估质量**(标杆 Hanover et al., L1-NMPC, RA-L 2021)。与主方法(MHE 显式估 m +
# 几何)构成机理对立的对照;本模块只实现估计律本身,纯 numpy、零 ROS/acados 依赖,
# 补偿注入方式(进 model.p vs 控制层叠加,决策点 D1)由 acados_nmpc_node 侧决定。
#
# 数学(L1 的三段解耦结构,自适应快慢与鲁棒性由不同旋钮独立控制——这是 L1 相对
# MRAC 的核心卖点):
#   1) 状态预测器(速度通道,3 维各自独立):
#        v̂̇ = a_nom + d̂ + A_s·(v̂ − v),   A_s = −a·I (Hurwitz, a>0)
#      a_nom = 标称模型在当前状态/输入下的平动加速度(外部算好传入,本模块不
#      持有动力学——保持与模型实现解耦,也方便单测用独立真值仿真对拍)。
#   2) 梯度型自适应律(Hovakimyan & Cao 书中 state-feedback L1 的标准律;
#      标量通道 P·b 归一并入 Γ):
#        d̂̇ = −Γ·ṽ,   ṽ = v̂ − v,   Γ = (a/2)²(默认,与预测器构成临界阻尼)
#      为什么不用 piecewise-constant 版:pc 律的设计目标是"下一拍预测误差归零"
#      (deadbeat),对常值扰动的稳态输出是 d·e^{−a·Ts}(a=10/dt=0.02 时偏低
#      18%,首版单测实测坐实)——在完整 L1 闭环里该偏差被补偿回路吸收,但本
#      项目要拿 d̂ 与 m_p·g 对表(Phase 1 验收<15%),不可接受。梯度律的 d̂
#      通道自带积分器(内模原理),常值扰动零稳态偏差。[ṽ; d̂−d] 的特征方程
#      s²+a·s+Γ=0,Γ=(a/2)² 时双实根 −a/2 无振荡。
#   3) 一阶低通(截止 omega_c [rad/s]):d̂_f = LPF(d̂)。补偿只用低频分量,高频
#      不确定性/噪声不进控制回路。**omega_c 是唯一的性能/鲁棒权衡旋钮**
#      (Phase 1 调参就扫它,见实施计划 §5)。
#
# 量纲约定:输入输出全部是**加速度(比力) [m/s²]**,不是力 [N]。理由:换算成力
# 需要乘质量,而"质量是多少"正是本对照实验里两个流派的分歧点——模块输出比力,
# 消费侧(NMPC 模型 vel_dot += d)不需要任何质量假设,对照才干净。
#
# dt 每次 update 传入而非构造时锁死:真实 ROS 定时器有抖动,离散系数(两个 exp,
# 3 维)每帧重算的开销可忽略,换来对时基抖动的正确性。

import numpy as np


class L1Augmentation:
    """平动 3-DoF lumped 扰动的 L1 自适应估计器(预测器+分段常值律+LPF)。"""

    def __init__(self, a_gain=10.0, omega_c=2.0, dim=3, gamma=None, d_max=20.0):
        # a_gain: 预测器收敛速率 A_s=−a·I。取 >> omega_c(快估计、慢补偿的
        #   时间尺度分离是 L1 结构成立的前提);默认 10 rad/s 对 50Hz 控制周期
        #   (a·dt=0.2、估计极点 −a/2 → λ·dt=0.1)欧拉离散化良好。
        # omega_c: 补偿低通截止 [rad/s]。默认 2 保守起步,Phase 1 扫 {2,5,10}。
        # gamma: 自适应增益 Γ;None=取 (a/2)²(与预测器临界阻尼,无振荡)。
        # d_max: d̂ 限幅 [m/s²](L1 里 Proj 算子的朴素版,防积分器在测量异常时
        #   漂走;默认 20≈2g,远大于任何合法载荷比力,正常工况不会触碰)。
        assert a_gain > 0.0 and omega_c > 0.0
        self.a = float(a_gain)
        self.gamma = float(gamma) if gamma is not None else (a_gain / 2.0) ** 2
        self.omega_c = float(omega_c)
        self.d_max = float(d_max)
        self.dim = int(dim)
        self.reset()

    def reset(self):
        """清零全部内部状态。attach 前初始化、drop 瞬间必须调用——掉包后真实
        lumped 扰动阶跃回零,残留的 d̂_f 会把推力拽偏(与 nmpc_node 里 drop 复位
        rate 增益是同一类必需处理,见 b3-strong-closed-loop-dr 的 drop 教训)。"""
        self.v_hat = None            # 预测器状态(首帧用实测速度初始化)
        self.d_hat = np.zeros(self.dim)    # 自适应律原始输出(未滤波)
        self.d_hat_f = np.zeros(self.dim)  # 低通后的补偿量(唯一对外输出)

    def update(self, v_meas, a_nom, dt):
        """推进一拍。
        v_meas: 实测速度 (dim,) [m/s](里程计)
        a_nom : 标称模型平动加速度 (dim,) [m/s²](含重力/推力/阻力,不含扰动)
        dt    : 本拍实际时长 [s]
        返回: d̂_f (dim,) [m/s²] — 滤波后的 lumped 扰动比力估计。
        """
        v_meas = np.asarray(v_meas, dtype=float)
        a_nom = np.asarray(a_nom, dtype=float)
        if self.v_hat is None:          # 首帧:预测器对齐实测,d̂ 从零起
            self.v_hat = v_meas.copy()
            return self.d_hat_f.copy()
        if dt <= 0.0:                   # 时基异常帧:不推进,保持上一拍输出
            return self.d_hat_f.copy()

        # 1) 梯度自适应律(d̂ 通道积分器→常值扰动无偏)+ 朴素 Proj 限幅
        v_tilde = self.v_hat - v_meas
        self.d_hat = np.clip(self.d_hat - dt * self.gamma * v_tilde,
                             -self.d_max, self.d_max)

        # 2) 预测器传播(欧拉;估计极点 −a/2,λ·dt≈0.1 时离散误差可忽略,
        #    A_s 项本身就是把 v_hat 往实测拉的稳定项)
        v_hat_dot = a_nom + self.d_hat - self.a * v_tilde
        self.v_hat = self.v_hat + dt * v_hat_dot

        # 3) 一阶低通(精确离散:alpha = 1-exp(-omega_c*dt),无条件稳定)
        alpha = 1.0 - np.exp(-self.omega_c * dt)
        self.d_hat_f = self.d_hat_f + alpha * (self.d_hat - self.d_hat_f)
        return self.d_hat_f.copy()
