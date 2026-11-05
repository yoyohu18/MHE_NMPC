#!/usr/bin/env python3
# 直来直去的往返(shuttle)轨迹，与本包圆形轨迹接口一致。用来直观验证机头朝向是否跟随飞行方向,
# 比圆形/8字更容易在 RViz 里一眼看出对不对。
#
# 参数化: x = d*sin(w*tc),沿 +X 轴在 [-d, d] 之间往复,w 决定往返速度和周期。
# 速度在两端(x=±d)自然降到 0、在中点(x=0)达到峰值 d*w,不需要额外的减速/
# 折返逻辑——跟圆形轨迹一样,纯粹靠三角函数的导数自动给出平滑的速度曲线。
#
# yaw 故意取常数 0(不跟随速度方向):往返轨迹里速度方向每半个周期就反一次号,
# 如果像圆形/8字那样让 yaw=atan2(vy,vx) 跟着速度方向算,会在折返瞬间让 yaw
# 指令瞬间跳 180°,这是任何飞控都不可能瞬间跟上的硬不连续点。固定 yaw=0 的
# 效果是:去程机头对着飞行方向(可以直接验证"前进时机头朝前"这件事),回程
# 机头不转身、保持朝 +X、整机倒着飞回来——这是四旋翼"全向推力"的正常特性,
# 不是 bug。

import numpy as np

from .common import euler_to_quat

from .acados_params import p


def build_reference_straight(t, d=2.0, w=0.3, z_hover=3.0,
                              hover_time=2.0, ramp_time=2.0):
    # t < hover_time: 在起点 (0,0,z_hover) 悬停(a=0 时 sin(a)=0,跟圆形/8字的
    # "起点即 a=0" 约定一致)
    if t < hover_time:
        pos = np.array([0.0, 0.0, z_hover])
        vel = np.zeros(3)
        q   = np.array([1.0, 0.0, 0.0, 0.0])
        om  = np.zeros(3)
        return np.concatenate([pos, vel, q, om])

    tc = t - hover_time

    if tc < ramp_time:
        s = tc / ramp_time
        alpha     = 10*s**3 - 15*s**4 + 6*s**5
        alpha_dot = (30*s**2 - 60*s**3 + 30*s**4) / ramp_time
    else:
        alpha     = 1.0
        alpha_dot = 0.0

    a = w * tc
    sa, ca = np.sin(a), np.cos(a)

    x_f  = d * sa
    vx_f = d * w * ca

    x0 = 0.0
    x  = x0 + alpha * (x_f - x0)
    vx = alpha_dot * (x_f - x0) + alpha * vx_f

    pos = np.array([x, 0.0, z_hover])
    vel = np.array([vx, 0.0, 0.0])
    q   = euler_to_quat(0.0, 0.0, 0.0)   # yaw 固定 0,理由见文件头注释
    om  = np.zeros(3)
    return np.concatenate([pos, vel, q, om])


def build_reference_window_straight(t_start, N, dt, d=2.0, w=0.3, z_hover=3.0):
    xref = np.zeros((p.nx, N + 1))
    for i in range(N + 1):
        xref[:, i] = build_reference_straight(t_start + i * dt, d, w, z_hover)
    return xref
