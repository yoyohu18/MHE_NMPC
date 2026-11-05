#!/bin/bash
# 主线 SITL 一键启动(headless):PX4 + Gazebo(磁吸夹爪世界)+ mavros + 电机转速桥 +
# 夹爪节点 + NMPC + 降阶 MHE。一次完整任务:起飞 → 对准 → 下降抓取 → 抬升 → 8 字 →
# 在 8 字尖端投放 → 释放确认(arm D:加速度计力平衡)→ 继续飞行。
#
# 主线参数全部写死在下面,取值与 ros2_ws_HJH 在 2026-11-05 主线下实际生效的参数逐项一致
# (录自运行中的批次进程)。只保留实验需要改动的开关:
#   PAYLOAD_KG            载荷质量 [kg]                0.30(0.15 / 0.30 已验证)
#   SPEED                 8 字峰值速度 [m/s]           2   (2 或 4;r=10 m 固定)
#   DETACH_DELAY_SEC      释放指令后物理脱离的延迟 [s] 0   (4 = 延迟脱离工况)
#   THRUST_FAULT_GAIN     估计/检测器侧推力映射增益    1.0 (0.9 / 1.1 = 标定误差测试)
#   UNCOMMANDED_LOSS      飞行中无指令掉包(故障注入)  false
#   DETACH_JAM            释放指令后夹爪卡死(故障注入) false
#   RESID_LOG_DIR         残差诊断 CSV 输出目录        空=不记录
#   RUN_DIR               日志目录                     <ws>/results
#
# ⚠️ 本脚本会 kill 所有同名仿真进程,并改写共享 PX4 目录里的世界文件。所以启动前检查:
#   有任何 PX4/Gazebo/节点进程在跑,或批次锁 /tmp/bodyrate_repeat.lock 被占用 → 拒绝启动。
#   批次驱动自己持锁调用本脚本时设 SITL_LOCK_HELD=1。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-/home/clear/PX4-Autopilot}"
PKG="$WS/src/offboard_test_acados"
GRIPPER_DIR="$PKG/gz_plugins/magnetic_gripper"
WORLD_SRC="$PKG/worlds/gripper/gripper_test.sdf"
PX4_WORLDS="$PX4_DIR/Tools/simulation/gz/worlds"
RUN_DIR="${RUN_DIR:-$WS/results}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOCK=/tmp/bodyrate_repeat.lock

# ---------------------------------------------------------------- 安全检查
STACK_PATTERNS=("px4_sitl_default/bin/px4" "gz sim" "make px4_sitl" "mavros/mavros_node"
  "lib/offboard_test_acados/acados_nmpc_node" "lib/offboard_test_acados/mhe_node"
  "lib/offboard_test_acados/proximity_gripper_node" "ros_gz_bridge" "gcs_heartbeat.py")
if [ "${SITL_LOCK_HELD:-0}" != 1 ]; then
  exec 8>"$LOCK"
  if ! flock -n 8; then
    echo "[run_sitl] REFUSED: batch lock $LOCK is held (an experiment is running)" >&2; exit 2
  fi
  for pat in "${STACK_PATTERNS[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
      echo "[run_sitl] REFUSED: a simulation process is already running ($pat)." >&2
      echo "           Stop it first (scripts/stop_sitl.sh) — this script would kill it." >&2
      exit 2
    fi
  done
  exec 8>&-
fi

# ---------------------------------------------------------------- 实验开关
PAYLOAD_KG="${PAYLOAD_KG:-0.30}"
SPEED="${SPEED:-2}"
DETACH_DELAY_SEC="${DETACH_DELAY_SEC:-0.0}"
THRUST_FAULT_GAIN="${THRUST_FAULT_GAIN:-1.0}"
UNCOMMANDED_LOSS="${UNCOMMANDED_LOSS:-false}"
DETACH_JAM="${DETACH_JAM:-false}"
case "$SPEED" in
  2) DYN_W=0.1415; DYN_RAMP=9.36 ;;
  4) DYN_W=0.283;  DYN_RAMP=0.0 ;;
  *) echo "[run_sitl] SPEED must be 2 or 4 (got $SPEED)" >&2; exit 2 ;;
esac
f() { python3 -c "print(float('$1'))"; }
PAYLOAD_KG=$(f "$PAYLOAD_KG"); DETACH_DELAY_SEC=$(f "$DETACH_DELAY_SEC")
THRUST_FAULT_GAIN=$(f "$THRUST_FAULT_GAIN")
BOX_I=$(python3 -c "print(f'{$PAYLOAD_KG * 0.00375:.6f}')")
ATTACH_TOL=0.03          # 夹爪吸附半径 = 偏心 0 + 容差

mkdir -p "$RUN_DIR"
# 记录给 provenance 的实验条件(白名单前缀 GRIP_/MHE_/EXP_ …)
export GRIP_PAYLOAD_KG="$PAYLOAD_KG" GRIP_SPEED="$SPEED" GRIP_DETACH_DELAY_SEC="$DETACH_DELAY_SEC" \
       MHE_THRUST_FAULT_GAIN="$THRUST_FAULT_GAIN" GRIP_UNCOMMANDED_LOSS="$UNCOMMANDED_LOSS" \
       GRIP_DETACH_JAM="$DETACH_JAM"
bash "$WS/scripts/tools/record_experiment_provenance.sh" "$RUN_DIR/provenance_$STAMP" "run_sitl:$STAMP"

# ---------------------------------------------------------------- 清理与构建
find "$RUN_DIR" -maxdepth 1 -type f -name 'px4_*.log' -delete 2>/dev/null || true
bash "$WS/scripts/stop_sitl.sh" --quiet
# PX4 会把 ParamSetV2 改过的参数落盘;每次干净启动前清掉,防止跨架污染
find "$PX4_DIR/build/px4_sitl_default/rootfs" -maxdepth 1 -name 'parameters*.bson' -delete 2>/dev/null || true

set +u; source /opt/ros/jazzy/setup.bash; set -u   # ROS setup 脚本引用未定义变量
(cd "$WS" && colcon build --packages-select offboard_test_acados >/dev/null) || {
  echo "[run_sitl] colcon build failed" >&2; exit 1; }
set +u; source "$WS/install/setup.bash"; set -u
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/clear/acados}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$GRIPPER_DIR/build"
(cd "$GRIPPER_DIR/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make >/dev/null) || {
  echo "[run_sitl] gripper plugin build failed" >&2; exit 1; }
export GZ_SIM_SYSTEM_PLUGIN_PATH="$GRIPPER_DIR/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

# 世界文件:箱子质量/惯量按本次载荷写入(写到 PX4 的 worlds 目录,PX4 只从那里加载)
sed -e "s|<mass>[0-9.]*</mass>|<mass>$PAYLOAD_KG</mass>|" \
    -e "s|<ixx>[0-9.]*</ixx>|<ixx>$BOX_I</ixx>|" \
    -e "s|<iyy>[0-9.]*</iyy>|<iyy>$BOX_I</iyy>|" \
    -e "s|<izz>[0-9.]*</izz>|<izz>$BOX_I</izz>|" \
    "$WORLD_SRC" > "$PX4_WORLDS/gripper_test.sdf"
PROX_YAML="$RUN_DIR/grip_prox_override_$STAMP.yaml"
cat > "$PROX_YAML" <<EOF
proximity_gripper_node:
  ros__parameters:
    r_xy: $ATTACH_TOL
    attach_dwell_sec: 0.0
    attach_r_min: 0.0
    h_min: 0.15
    h_max: 0.45
EOF

# ---------------------------------------------------------------- 启动
nohup bash -c "export GZ_SIM_SYSTEM_PLUGIN_PATH='$GZ_SIM_SYSTEM_PLUGIN_PATH' && \
    cd '$PX4_DIR' && HEADLESS=1 PX4_GZ_WORLD=gripper_test PX4_GZ_NO_FOLLOW=1 make px4_sitl gz_x500" 2>&1 \
    | ( head -c "${PX4_LOG_CAP:-5M}" > "$RUN_DIR/grip_px4_$STAMP.log"; cat >/dev/null ) &
sleep 18
nohup python3 "$WS/scripts/tools/gcs_heartbeat.py" > "$RUN_DIR/grip_gcs_hb_$STAMP.log" 2>&1 &
nohup ros2 launch mavros px4.launch fcu_url:=udp://:14540@ 2>&1 \
    | ( head -c "${MAVROS_LOG_CAP:-5M}" > "$RUN_DIR/grip_mavros_$STAMP.log"; cat >/dev/null ) &
sleep 8
nohup ros2 run ros_gz_bridge parameter_bridge \
    "/x500_0/command/motor_speed@actuator_msgs/msg/Actuators[gz.msgs.Actuators" \
    > "$RUN_DIR/grip_bridge_$STAMP.log" 2>&1 &
sleep 2
nohup ros2 run offboard_test_acados proximity_gripper_node --ros-args \
    --params-file "$PKG/config/gripper/gripper_params.yaml" \
    -p drone_model:=x500_0 -p r_xy:=$ATTACH_TOL -p h_min:=0.35 -p h_max:=0.60 \
    -p detach_delay_sec:=$DETACH_DELAY_SEC -p detach_jam:=$DETACH_JAM \
    --params-file "$PROX_YAML" \
    > "$RUN_DIR/grip_proximity_$STAMP.log" 2>&1 &
nohup gz topic -e -t /gripper/state > "$RUN_DIR/grip_state_$STAMP.log" 2>&1 &

NMPC_LOG="$RUN_DIR/grip_nmpc_$STAMP.log"
# 主线参数是节点默认值(acados_nmpc_node.py);这里只传实验开关。
nohup ros2 run offboard_test_acados acados_nmpc_node --ros-args \
    -p grip_dyn_w:=$DYN_W -p grip_dyn_ramp:=$DYN_RAMP \
    -p thrust_map_fault_gain:=$THRUST_FAULT_GAIN \
    -p grip_uncommanded_loss:=$UNCOMMANDED_LOSS \
    > "$NMPC_LOG" 2>&1 &

RESID_ARG=()
[ -n "${RESID_LOG_DIR:-}" ] && RESID_ARG=(-p "residual_log_dir:=$RESID_LOG_DIR")
MHE_LOG="$RUN_DIR/grip_mhe_$STAMP.log"
nohup ros2 run offboard_test_acados mhe_node --ros-args \
    --params-file "$PKG/config/gripper/gripper_params.yaml" "${RESID_ARG[@]}" \
    -p motor_speed_topic:=/x500_0/command/motor_speed \
    -p grip_payload_envelope:=0.3 -p eval_true_payload_mass:=$PAYLOAD_KG \
    -p thrust_map_fault_gain:=$THRUST_FAULT_GAIN \
    -p motor_window_avg:=true -p input_min_z:=0.5 \
    -p event_trigger_enable:=true -p confirm_thresh_alpha:=1.5 \
    -p no_payload_score_mode:=mass \
    > "$MHE_LOG" 2>&1 &

echo "MHE_NMPC SITL up: nmpc=$NMPC_LOG mhe=$MHE_LOG"
echo "  payload=${PAYLOAD_KG}kg speed=${SPEED}m/s detach_delay=${DETACH_DELAY_SEC}s thrust_gain=$THRUST_FAULT_GAIN"
echo "  stop with: bash $WS/scripts/stop_sitl.sh"
