#!/usr/bin/env python3
# 8 字轨迹(Gerono lemniscate)参考生成，与本包圆形轨迹接口一致。
#
# 参数化: x=r*sin(a), y=r*sin(a)*cos(a)=r/2*sin(2a), a=w*tc 匀角速度转动。
# 在交叉点(a=0,π,即原点)处 v=wr*(cos a, cos 2a) 幅值固定为 wr*sqrt(2)≠0,
# 且 ax,ay 都正比于 sin(a)/sin(2a),在 a=0,π 处恰好为 0 ——也就是说这条曲线
# 在交叉点速度不反转、曲率不会炸到无穷大(不是尖点),但 yaw 仍会在交叉点
# 前后转得比圆形轨迹快,跟踪难度天然更高。

import math

import numpy as np

from .common import euler_to_quat

from .acados_params import p


def auto_ramp_time(w, base=4.0):
    """按角速率 w 反推"起振不冲击"所需的 ramp_time,给高机动工况用。

    ramp 段的速度是 alpha_dot*(x_f-x0) + alpha*vx_f 两项之和。第一项是振幅
    渐增本身带来的**额外**速度,峰值 = (1.875/ramp_time)*r(5次 smooth-step 的
    alpha_dot 峰值是 1.875/ramp_time,x_f-x0 最大是 r);第二项是轨迹自身的
    速度,峰值 √2*r*w。要求额外项不超过轨迹特征速度的一半:

        1.875*r/ramp_time <= 0.5*√2*r*w   =>   ramp_time >= 2.65/w

    r 被约掉了——需要多长 ramp 只取决于 w,跟轨迹尺寸无关。注意这是**相对**
    判据:w 越小(轨迹越慢)反而要越长的 ramp,因为 alpha 的 smooth-step 按
    ramp_time 归一,而目标速度正比于 w。

    **不改任何现有默认值**:build_reference_figure8 的 ramp_time 仍是写死的
    4.0,调用方要自适应必须显式传 auto_ramp_time(w)。这样低速工况(r=1.0/
    w=0.3 裸机、r=0.8/w=0.25 gripper)的历史批次逐字节复现。
    """
    return max(float(base), 2.65 / float(w))


def build_reference_figure8(t, r=1.0, w=0.3, z_hover=3.0, dz=0.5,
                             hover_time=2.0, ramp_time=4.0, yaw_ramp=False):
    # t < hover_time: 在 a=0 对应的起点 (0,0,z_hover) 悬停(a=0 时 sin(a)=0,
    # 两叶交汇处高度就是 z_hover,跟 dz 无关)
    # ramp_time 内  : 用 5 次 smooth-step 把振幅从 0 渐增到目标
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
    sa, ca   = math.sin(a), math.cos(a)
    s2a, c2a = math.sin(2*a), math.cos(2*a)

    # z 用 sin(a) 而不是 cos(a):x 也是 sin(a),两者同符号同步变化,所以右侧
    # 叶子(x>0,a∈(0,π))整体抬高、左侧叶子(x<0,a∈(π,2π))整体压低,在原点
    # 交叉点(a=0,π)两叶在同一高度 z_hover 相接——这才是"一边高一边低"的
    # 立体 8 字。用 cos(a) 的话,高度只跟"穿越原点的次数"挂钩,跟左右叶子
    # 无关,达不到这个效果。
    x_f  = r * sa
    y_f  = 0.5 * r * s2a
    z_f  = z_hover + dz * sa
    vx_f = r * w * ca
    vy_f = r * w * c2a
    vz_f = dz * w * ca
    ax_f = -r * w * w * sa
    ay_f = -2.0 * r * w * w * s2a

    # 起点 (a=0): (0, 0, z_hover)
    x0, y0, z0 = 0.0, 0.0, z_hover

    x = x0 + alpha * (x_f - x0)
    y = y0 + alpha * (y_f - y0)
    z = z0 + alpha * (z_f - z0)
    vx = alpha_dot * (x_f - x0) + alpha * vx_f
    vy = alpha_dot * (y_f - y0) + alpha * vy_f
    vz = alpha_dot * (z_f - z0) + alpha * vz_f
    ax = alpha_dot * vx_f + alpha * ax_f
    ay = alpha_dot * vy_f + alpha * ay_f

    # 机头跟随速度方向。⚠️ 这里有一处**与 ramp 正交**的不连续:ramp 的 alpha 是
    # 同比缩放 vx/vy 的,`atan2(vy,vx)` 对 alpha 完全不敏感——方向从 ramp 第一帧
    # 起就已经是稳态切向。所以再长的 ramp 也平滑不了 yaw,加长 ramp_time 无效。
    # 真正的跳变来自下面的低速兜底:|v|<0.01 时 yaw 被强制成 0,一旦越过阈值就
    # 瞬间跳到 a=0 处的解析切向 atan2(r*w, r*w)=45°(实测 0°→44.84°,约 0.15s 内)。
    denom = vx*vx + vy*vy
    if denom > 1e-4:
        yaw_traj      = math.atan2(vy, vx)
        yaw_rate_traj = (vx*ay - vy*ax) / denom
    else:
        # 速度太小时 atan2 数值上不可靠,退回稳态轨迹的解析切向(它与 alpha 无关,
        # 因此在 ramp 段全程有定义),而不是给一个跟轨迹无关的常数 0。
        yaw_traj      = math.atan2(vy_f, vx_f)
        yaw_rate_traj = 0.0

    if yaw_ramp:
        # 单独给**方向**做插值:对"相对 hover 朝向(yaw=0)的角度差"套用同一条
        # 5 次 smooth-step。alpha=0 时精确保持 hover 的 yaw=0、alpha=1 时精确
        # 等于轨迹切向,中间沿最短路径转过去,于是 hover→ramp 边界连续。
        # yaw_rate 相应含两项:轨迹自身的转率(缩放)+ 方向插值本身的转率。
        d = math.atan2(math.sin(yaw_traj), math.cos(yaw_traj))  # wrap 到 [-pi,pi]
        yaw      = alpha * d
        yaw_rate = alpha * yaw_rate_traj + alpha_dot * d
    elif denom > 1e-4:
        yaw, yaw_rate = yaw_traj, yaw_rate_traj
    else:
        yaw, yaw_rate = 0.0, 0.0

    pos = np.array([x, y, z])
    vel = np.array([vx, vy, vz])
    q   = euler_to_quat(0.0, 0.0, yaw)
    om  = np.array([0.0, 0.0, yaw_rate])
    return np.concatenate([pos, vel, q, om])


def build_reference_window_figure8(t_start, N, dt, r=1.0, w=0.3, z_hover=3.0, dz=0.5,
                                    hover_time=2.0, ramp_time=4.0, yaw_ramp=False):
    # hover_time/ramp_time 必须跟单点版取同一组值,否则 solve 用的预测窗口和
    # 单点参考(起飞判据/RViz)会在 ramp 段错开。默认值跟单点版一致。
    xref = np.zeros((p.nx, N+1))
    for i in range(N+1):
        xref[:, i] = build_reference_figure8(t_start + i*dt, r, w, z_hover, dz,
                                              hover_time, ramp_time, yaw_ramp)
    return xref
