#!/usr/bin/env python3
"""把一次飞行录下的 odometry + 电机转速流离线喂给 mhe_node,记录它发布的全部帧。

用于新旧工作空间的等价性检查:分别用两个进程跑(两个包同名,不能同进程导入),
再用 compare_mhe_frames.py 比对。不需要 SITL,不启动 ROS spin:直接调用节点回调,
时钟替换为回放时钟,定时器按 0.1 s 节拍手动触发。

用法:
  PYTHONPATH=<ws>/src/offboard_test_acados python3 replay_mhe_node.py <resid_dir> <out.npz> [--old]
--old 表示被测的是 ros2_ws_HJH 的旧节点(需设 MHE_MODEL_VARIANT=reduced 等环境变量)。
"""
import argparse
import glob
import os
import sys

import numpy as np


def read_csv(path):
    a = np.genfromtxt(path, delimiter=',', names=True)
    return a


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        class _N:
            pass
        n = _N()
        n.nanoseconds = int(round(self.t * 1e9))
        return n


class _Rec:
    def __init__(self, clock):
        self.clock = clock
        self.rows = []

    def publish(self, msg):
        self.rows.append((self.clock.t, list(msg.data) if hasattr(msg.data, '__len__')
                          else [float(msg.data)]))


PARAMS = {   # 2026-11-05 主线 mhe_node 实际生效参数(录自运行中的批次进程)
    'motor_speed_topic': '/x500_0/command/motor_speed',
    'external_event_inputs': False, 'event_signal_mode': 'residual',
    'event_trigger_enable': True, 'schedule_theta': [-4.0, 0.0, 0.0, 0.0],
    'event_confirm_thresh_n': 1.5, 'confirm_thresh_alpha': 1.5,
    'grip_payload_envelope': 0.3, 'eval_true_payload_mass': 0.3,
    'geom_release_mode': 'self', 'release_resid_enable': True,
    'release_resid_floor_n': 0.9, 'release_resid_strong_persist': 4,
    'c_xy_mass_release_mp': 0.03, 'c_xy_mass_release_persist': 20,
    'c_xy_mass_arm_ratio': 3.0, 'c_xy_mass_arm_persist': 20,
    'c_xy_est_enable': True, 'c_xy_from_moment': True,
    'maneuver_gate_enable': False, 'resid_release_geom': True,
    'resid_confirm_enable': True, 'release_arm_on_command': False,
    'release_baseline': 'P', 'no_payload_score_mode': 'mass',
    'thrust_map_fault_gain': 1.0, 'motor_speed_noise_frac': 0.0,
    'resid_step_enable': False, 'payload_lost_watch_enable': False,
    'mhe_tau_source': 'phys_full', 'motor_window_avg': True,
    'reanchor_on_moment_overrange': True, 'pre_offboard_estimate': False,
    'pre_offboard_min_z': 0.5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('resid_dir')
    ap.add_argument('out')
    ap.add_argument('--old', action='store_true')
    ap.add_argument('--gain', type=float, default=1.0)
    a = ap.parse_args()

    import rclpy
    from rclpy.parameter import Parameter
    from nav_msgs.msg import Odometry
    from actuator_msgs.msg import Actuators

    params = dict(PARAMS, thrust_map_fault_gain=a.gain)
    args = ['--ros-args']
    for k, v in params.items():
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        elif isinstance(v, list):
            v = '[' + ','.join(str(x) for x in v) + ']'
        args += ['-p', f'{k}:={v}']
    rclpy.init(args=args)
    if a.old:
        from offboard_test_acados.mhe_node import MHENode
    else:
        from offboard_test_acados.mhe.node import MHENode

    clock = _Clock()
    od = read_csv(glob.glob(os.path.join(a.resid_dir, 'resid_odom_*.csv'))[0])
    mo = read_csv(glob.glob(os.path.join(a.resid_dir, 'resid_motor_*.csv'))[0])
    t0 = min(od['t_recv_ns'][0], mo['t_recv_ns'][0]) * 1e-9
    clock.t = t0
    node = MHENode()
    node.get_clock = lambda: clock
    pe = _Rec(clock)
    mp = _Rec(clock)
    node.payload_estimate_pub = pe
    node.mass_pub = mp
    node.tau_phys_pub = _Rec(clock)
    if hasattr(node, 'c_xy_est_pub'):
        node.c_xy_est_pub = _Rec(clock)
    node.timer.cancel()

    ev = [(t * 1e-9, 0, i) for i, t in enumerate(od['t_recv_ns'])]
    ev += [(t * 1e-9, 1, i) for i, t in enumerate(mo['t_recv_ns'])]
    ev.sort()
    t_end = ev[-1][0]
    next_tick = t0 + 0.1
    k = 0
    while k < len(ev) or next_tick <= t_end:
        if k < len(ev) and ev[k][0] < next_tick:
            t, kind, i = ev[k]
            clock.t = t
            if kind == 0:
                m = Odometry()
                m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (
                    float(od['px'][i]), float(od['py'][i]), float(od['pz'][i]))
                o = m.pose.pose.orientation
                o.w, o.x, o.y, o.z = (float(od[c][i]) for c in ('qw', 'qx', 'qy', 'qz'))
                tl = m.twist.twist.linear
                tl.x, tl.y, tl.z = (float(od[c][i]) for c in ('vb_x', 'vb_y', 'vb_z'))
                ta = m.twist.twist.angular
                ta.x, ta.y, ta.z = (float(od[c][i]) for c in ('wb_x', 'wb_y', 'wb_z'))
                node.odom_cb(m)
            else:
                m = Actuators()
                m.velocity = [float(mo[c][i]) for c in ('w1', 'w2', 'w3', 'w4')]
                node.motor_speed_cb(m)
            k += 1
        else:
            clock.t = next_tick
            node.timer_cb()
            next_tick += 0.1
    np.savez(a.out,
             pe_t=np.array([r[0] for r in pe.rows]),
             pe=np.array([r[1] for r in pe.rows]),
             mp_t=np.array([r[0] for r in mp.rows]),
             mp=np.array([r[1][0] for r in mp.rows]) if mp.rows else np.zeros(0))
    print(f'frames={len(pe.rows)} mass_msgs={len(mp.rows)}', file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
