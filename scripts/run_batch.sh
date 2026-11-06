#!/bin/bash
# 主线 SITL 批次驱动:随机化排程 → 逐架 run_sitl.sh → 判定 → 停栈 → manifest → 汇总。
#
# 每架就是一次 scripts/run_sitl.sh(部署配置:降阶 MHE R + 力平衡释放 arm D),只改
# run_sitl.sh 的实验开关。所有日志写进该架自己的目录(RUN_DIR=flights/<idx>_<cell>)。
#
# 实验(EXP),工况 c1 0.15kg@2m/s  c2 0.15@4  c3 0.30@4  c4 0.30@2:
#   smoke         2 架:c4 正常投放 + c1 意外脱落(检验驱动本身)
#   release       脱离 {立即 d0, 延迟 4 s d4} × c1–c4 × 9 轮                  = 72
#   faults        {意外脱落 loss, 夹爪卡死 jam} × c1–c4 × 6 轮                 = 48
#   thrust        推力增益 {0.90, 1.10} × c1–c4 × 6 轮,立即脱离               = 48
#   thrust_small  推力增益 {0.95, 1.05} × c1–c4 × 3 轮,立即脱离               = 24
#   thrust_delay  推力增益 {0.90, 1.10} × c1–c4 × 3 轮,脱离延迟 4 s            = 24
# 每轮(block)内全部格子随机排序,排程与种子开跑前一次写死在 schedule.csv。
# 推力增益 = run_sitl.sh 的 THRUST_FAULT_GAIN:**同时**作用在 MHE 与加速度计检测器
# 的推力映射上(一个标错的 k_f 会同时影响两者);被控对象与执行器油门映射不变。
#
# 补跑规则(跑前冻结):不按结果补跑。只有基础设施失败(栈未起/无 NMPC 日志、日志缺失、
# 配置与排程不符、物理吸附失败)在主排程结束后补跑**一次**,补跑行 kind=rerun、
# replaces=<原 idx> 追加进 schedule/manifest,原架次保留。发散、求解失败、漏检、
# UNRESOLVED、提前确认都是结果。
#
# 用法:
#   EXP=faults bash scripts/run_batch.sh                       # 新批次(目录 results/batch_<EXP>_<时间>)
#   BATCH_DIR=<已有批次目录> bash scripts/run_batch.sh          # 续跑(manifest 已有的 idx 跳过)
#   DRY_RUN=1 EXP=faults bash scripts/run_batch.sh             # 只打印排程
#   SEED=<整数>                                                 # 指定排程种子(默认取时间)
#
# ⚠️ 批中禁止改运行时代码:每架开跑前核对代码指纹,变化即停批(BLOCKED)。
# ⚠️ 与 ros2_ws_HJH 共用批次锁 /tmp/bodyrate_repeat.lock,任一边在跑另一边都拒绝启动。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$WS/scripts/run_batch.sh"
AGG="$WS/scripts/tools/batch_aggregate.py"
LOCK=/tmp/bodyrate_repeat.lock
POST_SEC=${POST_SEC:-30}            # 终态事件后继续观察的秒数
TRIG_TIMEOUT=${TRIG_TIMEOUT:-420}   # 起飞后等投放指令/注入的上限 [s](正常约 130 s)
RERUN_INFRA=${RERUN_INFRA:-1}

if [ -n "${BATCH_DIR:-}" ]; then
  BDIR=$BATCH_DIR
  EXP=$(sed -n 's/^experiment=//p' "$BDIR/batch_info.txt" 2>/dev/null)
  [ -n "$EXP" ] || { echo "[batch] $BDIR has no batch_info.txt"; exit 2; }
else
  EXP=${EXP:?EXP=smoke|release|faults|thrust|thrust_small|thrust_delay}
  BDIR=$WS/results/batch_${EXP}_$(date +%Y%m%d_%H%M%S)
fi
MAN=$BDIR/manifest.csv
SCHED=$BDIR/schedule.csv

work_args() {   # 工况 -> run_sitl.sh 的载荷/速度
  case "$1" in
    c1) echo "PAYLOAD_KG=0.15 SPEED=2" ;;
    c2) echo "PAYLOAD_KG=0.15 SPEED=4" ;;
    c3) echo "PAYLOAD_KG=0.30 SPEED=4" ;;
    c4) echo "PAYLOAD_KG=0.30 SPEED=2" ;;
    *)  echo "BAD_WORK" ;;
  esac
}
row_args() {    # work scen fault tgain
  local d j l
  case "$2" in d0) d=0.0 ;; d4) d=4.0 ;; *) d=BAD_SCEN ;; esac
  case "$3" in none) l=false; j=false ;; loss) l=true; j=false ;; jam) l=false; j=true ;;
               *) l=BAD_FAULT; j=BAD_FAULT ;; esac
  echo "$(work_args "$1") DETACH_DELAY_SEC=$d THRUST_FAULT_GAIN=$4 UNCOMMANDED_LOSS=$l DETACH_JAM=$j"
}

# 运行时代码指纹:飞行栈包源码与配置、世界与夹爪插件、启动/停止脚本、用到的工具、本驱动。
fingerprint() {
  ( cd "$WS" && {
      find src/offboard_test_acados -type f \
        \( -name '*.py' -o -name '*.yaml' -o -name '*.json' -o -name '*.sdf' \
           -o -name '*.cc' -o -name '*.hh' -o -name 'CMakeLists.txt' -o -name 'package.xml' \
           -o -name 'setup.py' -o -name 'setup.cfg' \) \
        -not -path '*__pycache__*' -not -path '*acados_generated_files*' \
        -not -path '*/build/*' -not -path '*/test/*'
      echo scripts/run_sitl.sh; echo scripts/stop_sitl.sh; echo scripts/run_batch.sh
      echo scripts/tools/gcs_heartbeat.py; echo scripts/tools/record_experiment_provenance.sh
    } | sort | xargs sha256sum ) | sha256sum | cut -c1-16
}

stop_stack() { SITL_LOCK_HELD=1 bash "$WS/scripts/stop_sitl.sh" --quiet 9>&- 3<&-; sleep 6; }

# ---- 排程 ----
gen_schedule() {
  mkdir -p "$BDIR"
  local seed=${SEED:-$(date +%s)}
  EXP="$EXP" python3 - "$SCHED" "$seed" <<'EOF'
import csv, os, random, sys
out, seed = sys.argv[1], int(sys.argv[2])
exp = os.environ['EXP']
rng = random.Random(seed)
W = ('c1', 'c2', 'c3', 'c4')
DESIGN = {   # exp -> (rounds, [(scen, fault, tgain), ...])
    'release':      (9, [('d0', 'none', '1.0'), ('d4', 'none', '1.0')]),
    'faults':       (6, [('d0', 'loss', '1.0'), ('d0', 'jam', '1.0')]),
    'thrust':       (6, [('d0', 'none', '0.90'), ('d0', 'none', '1.10')]),
    'thrust_small': (3, [('d0', 'none', '0.95'), ('d0', 'none', '1.05')]),
    'thrust_delay': (3, [('d4', 'none', '0.90'), ('d4', 'none', '1.10')]),
}
rows = []    # (block, cell_id, work, scen, fault, tgain, kind)
if exp == 'smoke':
    rows = [(0, 'c4_d0_none_k1.0', 'c4', 'd0', 'none', '1.0', 'smoke'),
            (0, 'c1_d0_loss_k1.0', 'c1', 'd0', 'loss', '1.0', 'smoke')]
elif exp in DESIGN:
    rounds, levels = DESIGN[exp]
    cells = [(w, *lv) for w in W for lv in levels]
    for b in range(1, rounds + 1):
        cc = cells[:]
        rng.shuffle(cc)
        for w, sc, fa, tg in cc:
            rows.append((b, f'{w}_{sc}_{fa}_k{tg}', w, sc, fa, tg, 'main'))
else:
    sys.exit(f'unknown EXP {exp}')
with open(out, 'w', newline='') as f:
    wr = csv.writer(f, lineterminator='\n')
    wr.writerow(['idx', 'block', 'cell_id', 'work', 'scen', 'fault', 'tgain', 'kind', 'replaces'])
    for i, r in enumerate(rows, 1):
        wr.writerow([i, *r, ''])
print(f'[sched] {exp}: {len(rows)} rows, seed={seed}')
EOF
  echo "$seed" > "$BDIR/seed.txt"
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  BDIR=$(mktemp -d); SCHED=$BDIR/schedule.csv; gen_schedule; cat "$SCHED"; rm -rf "$BDIR"; exit 0
fi

# ---- 起跑前守卫 ----
exec 9>"$LOCK"
flock -n 9 || { echo "[batch] another batch holds $LOCK"; exit 1; }
for pat in "px4_sitl_default/bin/px4" "gz sim" "mavros/mavros_node" \
           "lib/offboard_test_acados/acados_nmpc_node" "lib/offboard_test_acados/mhe_node"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "[batch] refuse: a simulation process is already running ($pat)"; exit 2
  fi
done
trap 'echo "[batch] interrupted"; stop_stack; exit 130' INT TERM

FP_NOW=$(fingerprint)
if [ ! -f "$SCHED" ]; then
  gen_schedule
  echo "idx,block,cell_id,work,scen,fault,tgain,kind,replaces,stamp,started,finished,trigger_seen,terminal,diverged,fingerprint,flight_dir" > "$MAN"
  {
    echo "experiment=$EXP"
    echo "dir=$BDIR"
    echo "started=$(date +%FT%T%z)"
    echo "seed=$(cat "$BDIR/seed.txt")"
    echo "fingerprint=$FP_NOW"
    echo "git_head=$(git -C "$WS" rev-parse HEAD 2>/dev/null)"
    echo "git_dirty_runtime=$(git -C "$WS" status --porcelain -- src scripts | wc -l)"
    echo "post_sec=$POST_SEC trig_timeout=$TRIG_TIMEOUT rerun_infra=$RERUN_INFRA"
    echo "system=deployed (reduced MHE R + force-balance release, arm D); run_sitl.sh defaults"
  } > "$BDIR/batch_info.txt"
  git -C "$WS" status --porcelain > "$BDIR/git_status.txt" 2>/dev/null
  git -C "$WS" diff > "$BDIR/git_diff.patch" 2>/dev/null
fi
FP_FROZEN=$(sed -n 's/^fingerprint=//p' "$BDIR/batch_info.txt")
if [ "$FP_NOW" != "$FP_FROZEN" ]; then
  echo "[batch] BLOCKED: code fingerprint $FP_NOW != frozen $FP_FROZEN" | tee -a "$BDIR/blocked.txt"
  exit 3
fi

nsf() { local c; c=$(grep -c 'solve failed' "$1" 2>/dev/null); echo "${c:-0}"; }

run_one() {
  local idx=$1 block=$2 cid=$3 work=$4 scen=$5 fault=$6 tg=$7 kind=$8 rep=$9
  local args fdir t0 N stamp trig=0 term=none div=0
  args=$(row_args "$work" "$scen" "$fault" "$tg")
  case "$args" in *BAD_*) echo "[batch] bad cell idx=$idx"; exit 2 ;; esac
  if [ "$(fingerprint)" != "$FP_FROZEN" ]; then
    echo "[batch] BLOCKED at idx=$idx: code fingerprint changed" | tee -a "$BDIR/blocked.txt"; exit 3
  fi
  fdir="$BDIR/flights/${idx}_${cid}"
  mkdir -p "$fdir/resid"
  echo "$args" | tr ' ' '\n' > "$fdir/args.txt"
  t0=$(date +%FT%T)
  echo "[batch] === $EXP idx=$idx block=$block $cid kind=$kind ($(date +%T)) ==="
  stop_stack
  # run_sitl.sh 起完栈即返回;输出写文件(不能用 $(...) 捕获:后台进程继承了管道)。
  env $args SITL_LOCK_HELD=1 RUN_DIR="$fdir" RESID_LOG_DIR="$fdir/resid" \
    bash "$WS/scripts/run_sitl.sh" > "$fdir/launch.log" 2>&1 9>&- 3<&- < /dev/null
  N=$(grep -oE 'nmpc=[^ ]+' "$fdir/launch.log" | head -1 | cut -d= -f2)
  if [ -z "$N" ] || [ ! -e "$N" ]; then
    echo "$idx,$block,$cid,$work,$scen,$fault,$tg,$kind,$rep,NA,$t0,$(date +%FT%T),0,none,0,$FP_FROZEN,$fdir" >> "$MAN"
    echo "[batch] idx=$idx startup failed (see $fdir/launch.log)"
    stop_stack; return
  fi
  stamp=$(basename "$N" .log); stamp=${stamp#grip_nmpc_}
  local TRIG="DROP command issued|UNCOMMANDED LOSS injected"
  local TERM="DROP complete|UNRESOLVED|LOST: PAYLOAD LOST"
  for _ in $(seq 1 $((TRIG_TIMEOUT / 5))); do
    grep -qE "$TRIG" "$N" 2>/dev/null && { trig=1; break; }
    grep -q "LOST: PAYLOAD LOST" "$N" 2>/dev/null && break
    [ "$(nsf "$N")" -gt 100 ] && { div=1; break; }
    pgrep -f "lib/offboard_test_acados/acados_nmpc_node" >/dev/null || {
      sleep 5; pgrep -f "lib/offboard_test_acados/acados_nmpc_node" >/dev/null || break; }
    sleep 5
  done
  if [ "$div" = 0 ]; then
    for _ in $(seq 1 45); do
      grep -qE "$TERM" "$N" 2>/dev/null && break
      [ "$(nsf "$N")" -gt 100 ] && { div=1; break; }
      sleep 1
    done
    term=$(grep -oE "$TERM" "$N" 2>/dev/null | head -1 | awk '{print $1"_"$2}')
    [ -z "$term" ] && term=none
    for _ in $(seq 1 "$POST_SEC"); do
      [ "$(nsf "$N")" -gt 100 ] && { div=1; break; }
      sleep 1
    done
  fi
  [ "$(nsf "$N")" -gt 100 ] && div=1
  stop_stack
  echo "$idx,$block,$cid,$work,$scen,$fault,$tg,$kind,$rep,$stamp,$t0,$(date +%FT%T),$trig,$term,$div,$FP_FROZEN,$fdir" >> "$MAN"
  echo "[batch] idx=$idx stamp=$stamp trig=$trig term=$term div=$div"
}

run_pending() {
  local last_idx
  last_idx=$(tail -1 "$SCHED" | cut -d, -f1)
  while IFS=, read -r -u 3 idx block cid work scen fault tg kind rep; do
    [ "$idx" = idx ] && continue
    rep=${rep%$'\r'}
    grep -q "^$idx," "$MAN" && continue
    echo "[batch] progress $idx/$last_idx"
    run_one "$idx" "$block" "$cid" "$work" "$scen" "$fault" "$tg" "$kind" "$rep" < /dev/null
    if [ $((idx % 8)) = 0 ]; then
      python3 "$AGG" "$BDIR" --quiet >> "$BDIR/aggregate.log" 2>&1 || true
    fi
  done 3< "$SCHED"
}

run_pending
if [ "$EXP" != smoke ] && [ "$RERUN_INFRA" = 1 ] && ! grep -q ',rerun,' "$SCHED"; then
  n=$(python3 "$AGG" "$BDIR" --emit-infra-reruns 2>>"$BDIR/aggregate.log" | tail -1)
  echo "[batch] infra rerun pass: ${n:-0} rows appended"
  [ "${n:-0}" != 0 ] && run_pending
fi
stop_stack
python3 "$AGG" "$BDIR" >> "$BDIR/aggregate.log" 2>&1 || true
echo "[batch] $EXP complete: $BDIR/summary.md"
