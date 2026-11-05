#!/usr/bin/env python3
"""无头 SITL 用的 GCS 心跳器(顶替 QGC 满足 PX4 解锁健康检查):
监听 QGC 默认口 14550,等到 PX4 心跳后以 GCS 身份 1Hz 回心跳。
由 run_sitl_headless.sh 启动;独立文件是为了进程名可被 teardown pkill 匹配。"""
import time

from pymavlink import mavutil

m = mavutil.mavlink_connection('udpin:0.0.0.0:14550',
                               source_system=255, source_component=190)
m.wait_heartbeat()
print('PX4 heartbeat seen, sending GCS heartbeats', flush=True)
while True:
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    time.sleep(1.0)
