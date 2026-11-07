#!/bin/bash
# 停掉 run_sitl_viz.sh 起的可视化进程(RViz、robot_state_publisher、scripts/viz 辅助节点),
# 再调用 stop_sitl.sh 停主线仿真栈(Gazebo GUI 客户端 `gz sim -g` 由它的 "gz sim" 匹配收掉)。
# 批次锁被占用时 stop_sitl.sh 会拒绝执行;这里同样先检查,避免只杀一半。
# 用法: bash scripts/stop_sitl_viz.sh [--quiet]
WS="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/tmp/bodyrate_repeat.lock
exec 8>"$LOCK"
flock -n 8 || { echo "[stop_sitl_viz] REFUSED: batch lock $LOCK is held (an experiment is running)" >&2; exit 2; }
exec 8>&-
for pat in "rviz2 -d .*nmpc_view_gripper" "robot_state_publisher.*robot name=.x500." \
           "scripts/viz/drone_tf_broadcaster.py" "scripts/viz/prop_joint_state_publisher.py"; do
  for pp in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pp" = "$$" ] && continue
    kill -9 "$pp" 2>/dev/null || true
  done
done
bash "$WS/scripts/stop_sitl.sh" "$@"
