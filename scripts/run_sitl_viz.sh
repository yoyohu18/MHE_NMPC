#!/bin/bash
# 主线 SITL 可视化启动:在 scripts/run_sitl.sh(主线配置,未改动)之上叠加
#   Gazebo GUI 客户端(接到同一个 headless server,看物理:抓取/抬升/8 字/投放)
#   RViz2(控制端:8 字参考 vs 实际轨迹 + NMPC 预测 horizon + x500 机体模型 + 旋翼动画)
# 迁移自 ros2_ws_HJH/src/scripts/gripper/run_sitl_gripper_viz.sh。旧脚本自带一整套
# 历史参数(θ* 调度、偏心、事件/几何档、力矩来源 …),主线已全部写死或删除;这里
# **不重复任何控制/估计参数**,飞的就是 run_sitl.sh / run_batch.sh 的那套配置。
# 可视化辅助节点与 URDF 在 scripts/viz/,不在 ROS 包里,不进代码指纹。
#
# 实验开关与 run_sitl.sh 完全相同(原样透传):
#   PAYLOAD_KG SPEED DETACH_DELAY_SEC THRUST_FAULT_GAIN UNCOMMANDED_LOSS DETACH_JAM RESID_LOG_DIR
# 可视化专用:
#   RUN_DIR        日志目录,默认 results/viz_<时间>(与实验日志分开)
#   RVIZ_CONFIG    RViz 配置,默认 config/gripper/nmpc_view_gripper_hifly.rviz
#   GZ_GUI=0       不开 Gazebo GUI(只看 RViz)
#   RVIZ=0         不开 RViz
#
# 用法:
#   bash scripts/run_sitl_viz.sh
#   PAYLOAD_KG=0.15 SPEED=4 DETACH_DELAY_SEC=4 bash scripts/run_sitl_viz.sh
#   bash scripts/stop_sitl_viz.sh                       # 收栈(含 RViz 与辅助节点)
#
# ⚠️ GUI 渲染会占 CPU/GPU,可能改变实时时序:可视化飞行只用于观察和录屏,
#    不作为实验数据。安全检查(批次锁、已有仿真进程)沿用 run_sitl.sh。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-/home/clear/PX4-Autopilot}"
PKG="$WS/src/offboard_test_acados"
VIZ="$WS/scripts/viz"
RVIZ_CONFIG="${RVIZ_CONFIG:-$PKG/config/gripper/nmpc_view_gripper_hifly.rviz}"
URDF_FILE="$VIZ/x500.urdf"
STAMP=$(date +%Y%m%d_%H%M%S)
export RUN_DIR="${RUN_DIR:-$WS/results/viz_$STAMP}"
GZ_GUI="${GZ_GUI:-1}"
RVIZ="${RVIZ:-1}"

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo "[run_sitl_viz] REFUSED: no DISPLAY (run from a desktop session, or use run_sitl.sh)" >&2
  exit 2
fi
for f in "$RVIZ_CONFIG" "$URDF_FILE" "$VIZ/drone_tf_broadcaster.py" "$VIZ/prop_joint_state_publisher.py"; do
  [ -f "$f" ] || { echo "[run_sitl_viz] missing $f" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------- Gazebo GUI 客户端
# run_sitl.sh 以 HEADLESS=1 起 PX4,PX4 只起 server(gz sim -s)。这里等 server 的
# /world/*/clock 出现后,用 PX4 同一套环境(gz_env.sh + GZ_IP)起 `gz sim -g` 接上去。
# 在本脚本的子 shell 里等待(命令行不含 "gz sim",不会被 run_sitl.sh 开头的清理误杀)。
gz_gui_watch() {
  set +u      # 子 shell 内:ROS setup 与 PX4 gz_env.sh 都引用未定义变量
  source /opt/ros/jazzy/setup.bash
  [ -f "$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh" ] && \
    . "$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh"
  export GZ_IP=127.0.0.1
  sleep 5                                   # 让 run_sitl.sh 先做完安全检查与清理
  for _ in $(seq 1 60); do
    if timeout 5 gz topic -l 2>/dev/null | grep -q '^/world/.*/clock'; then
      nohup gz sim -g > "$RUN_DIR/viz_gzgui_$STAMP.log" 2>&1 &
      echo "[run_sitl_viz] Gazebo GUI attached"
      return 0
    fi
    sleep 2
  done
  echo "[run_sitl_viz] WARN: Gazebo server not seen within ~2 min; GUI not started" >&2
}
[ "$GZ_GUI" = 1 ] && gz_gui_watch &

# ---------------------------------------------------------------- 主线栈(未改动)
bash "$WS/scripts/run_sitl.sh"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[run_sitl_viz] run_sitl.sh failed (rc=$rc); viz not started" >&2
  kill %1 2>/dev/null
  exit $rc
fi

# ---------------------------------------------------------------- RViz 与辅助节点
set +u; source /opt/ros/jazzy/setup.bash; set -u
if [ "$RVIZ" = 1 ]; then
  nohup ros2 run robot_state_publisher robot_state_publisher \
      --ros-args -p robot_description:="$(cat "$URDF_FILE")" \
      > "$RUN_DIR/viz_rsp_$STAMP.log" 2>&1 &
  nohup python3 "$VIZ/drone_tf_broadcaster.py" > "$RUN_DIR/viz_tf_$STAMP.log" 2>&1 &
  nohup python3 "$VIZ/prop_joint_state_publisher.py" --ros-args \
      -p motor_speed_topic:=/x500_0/command/motor_speed \
      > "$RUN_DIR/viz_props_$STAMP.log" 2>&1 &
  nohup rviz2 -d "$RVIZ_CONFIG" > "$RUN_DIR/viz_rviz_$STAMP.log" 2>&1 &
fi
wait %1 2>/dev/null   # 等 GUI 挂接结束(或超时)再返回

echo "MHE_NMPC SITL viz up: logs in $RUN_DIR"
echo "  Gazebo GUI=$GZ_GUI  RViz=$RVIZ ($RVIZ_CONFIG)"
echo "  stop with: bash $WS/scripts/stop_sitl_viz.sh"
