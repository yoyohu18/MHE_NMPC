#!/bin/bash
# 停掉本机所有主线仿真进程(PX4、Gazebo、mavros、桥接、三个节点、心跳)。
# ⚠️ 进程按名字匹配,与 ros2_ws_HJH 的同名进程无法区分:批次锁被占用时拒绝执行,
#    防止误杀正在运行的实验。批次驱动自己持锁调用时设 SITL_LOCK_HELD=1。
# 用法: bash stop_sitl.sh [--quiet]
LOCK=/tmp/bodyrate_repeat.lock
if [ "${SITL_LOCK_HELD:-0}" != 1 ]; then
  exec 8>"$LOCK"
  flock -n 8 || { echo "[stop_sitl] REFUSED: batch lock $LOCK is held (an experiment is running)" >&2; exit 2; }
  exec 8>&-
fi
for pat in "px4_sitl_default/bin/px4" "gz sim" "/gz " "ruby" "mavros/mavros_node" \
           "transport13/gz-transport-topic" \
           "lib/offboard_test_acados/proximity_gripper_node" \
           "lib/offboard_test_acados/acados_nmpc_node" \
           "lib/offboard_test_acados/mhe_node" \
           "ros_gz_bridge" "gcs_heartbeat.py" "ninja gz_x500" "make px4_sitl" "MicroXRCEAgent"; do
  for pp in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pp" = "$$" ] && continue
    kill -9 "$pp" 2>/dev/null || true
  done
done
sleep 2
[ "${1:-}" = "--quiet" ] || echo "[stop_sitl] simulation processes stopped"
