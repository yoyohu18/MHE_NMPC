#!/bin/bash
# 为每次实验冻结可审计的源码身份。
#
# 用法: record_experiment_provenance.sh <output-dir> <run-label>
#
# exact commit 只在所有运行时代码仓都 clean 时允许声称。dirty 时除了
# HEAD 还冻结 binary patch、未跟踪文件内容和 SHA-256，但在报告中仍明示
# exact_commit_claim=PROHIBITED，不再把“记了 HEAD”误写成“由该 commit 运行”。
set -euo pipefail

OUT_DIR="${1:?usage: $0 <output-dir> <run-label>}"
RUN_LABEL="${2:-experiment}"
WS="${EXPERIMENT_WS:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "$OUT_DIR"

# 只记录实验配置白名单，严禁把 token/凭据随整份 env 写进日志。
env | LC_ALL=C sort | awk -F= '
  $1 ~ /^(GRIP|MHE|NMPC|DROP|RELEASE|DJ|TRAJ|L1|XI|OMEGA|ATTACH|EVAL|CONTINUOUS|USE_MHE|PUBLISH|GEOM|RESID|EXP|J|ACC|NO_PAYLOAD)_/ {
    print
  }
' > "$OUT_DIR/experiment.env"

# experiment.env 只有显式设置的变量;代码默认值不落盘会让"改默认值"前后的批次
# 外观相同(09-15 种子默认开时暴露,见实验计划 §12.2)。这里再记每个开关的生效值。
python3 "$WS/scripts/tools/dump_effective_env.py" \
  "$WS/src/offboard_test_acados/offboard_test_acados" \
  > "$OUT_DIR/experiment.effective_env.tsv" \
  || echo "[provenance] WARN: effective env dump failed" >&2

printf '%s\n' \
  "schema_version=1" \
  "run_label=$RUN_LABEL" \
  "captured_at=$(date --iso-8601=seconds)" \
  "hostname=$(hostname)" \
  "kernel=$(uname -srmo)" \
  "ros_distro=${ROS_DISTRO:-unknown}" \
  "python=$(python3 --version 2>&1)" \
  > "$OUT_DIR/run.txt"

capture_repo() {
  local name="$1" repo="$2"
  local meta="$OUT_DIR/$name.git.txt"
  if [ ! -d "$repo" ] || ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
    printf '%s\n' "repo=$repo" "available=false" > "$meta"
    return
  fi

  git -C "$repo" -c core.quotepath=off status --porcelain=v1 --untracked-files=all \
    > "$OUT_DIR/$name.status"
  git -C "$repo" diff --binary --submodule=diff HEAD -- \
    > "$OUT_DIR/$name.tracked.patch"

  # paper/data 是论文 artifact，不参与仿真执行；尚未提交时不应在
  # 每轮溯源包里重复复制约 48 MiB。该排除会明记在 repo metadata 中。
  git -C "$repo" -c core.quotepath=off ls-files --others --exclude-standard -- \
    | awk '$0 !~ /^paper\/data\//' \
    > "$OUT_DIR/$name.untracked.files"
  : > "$OUT_DIR/$name.untracked.sha256"
  if [ -s "$OUT_DIR/$name.untracked.files" ]; then
    while IFS= read -r path; do
      (cd "$repo" && sha256sum -- "$path")
    done < "$OUT_DIR/$name.untracked.files" \
      > "$OUT_DIR/$name.untracked.sha256"
  fi
  # clean 仓也生成空归档，使每次记录的 schema 和验证流程完全一致。
  tar -czf "$OUT_DIR/$name.untracked.tar.gz" -C "$repo" \
    -T "$OUT_DIR/$name.untracked.files"

  local dirty_count dirty
  dirty_count=$(wc -l < "$OUT_DIR/$name.status")
  dirty=false
  [ "$dirty_count" -eq 0 ] || dirty=true
  printf '%s\n' \
    "repo=$repo" \
    "available=true" \
    "head=$(git -C "$repo" rev-parse HEAD)" \
    "branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)" \
    "describe=$(git -C "$repo" describe --always --dirty --tags 2>/dev/null || true)" \
    "commit_time=$(git -C "$repo" show -s --format=%cI HEAD)" \
    "dirty=$dirty" \
    "dirty_entries=$dirty_count" \
    "tracked_patch_sha256=$(sha256sum "$OUT_DIR/$name.tracked.patch" | awk '{print $1}')" \
    "untracked_manifest_sha256=$(sha256sum "$OUT_DIR/$name.untracked.sha256" | awk '{print $1}')" \
    "untracked_archive_sha256=$(sha256sum "$OUT_DIR/$name.untracked.tar.gz" | awk '{print $1}')" \
    "excluded_runtime_irrelevant=paper/data/**" \
    > "$meta"
  git -C "$repo" submodule status --recursive \
    > "$OUT_DIR/$name.submodules" 2>/dev/null || true
}

capture_repo workspace "$WS"
capture_repo px4 "${PX4_DIR:-/home/clear/PX4-Autopilot}"
capture_repo acados "${ACADOS_SOURCE_DIR:-/home/clear/acados}"

claim=ALLOWED
reason=all_runtime_source_repositories_clean
for meta in "$OUT_DIR"/*.git.txt; do
  if ! grep -qx 'available=true' "$meta" || ! grep -qx 'dirty=false' "$meta"; then
    claim=PROHIBITED
    reason=missing_or_dirty_runtime_source_repository
    break
  fi
done
printf '%s\n' \
  "exact_commit_claim=$claim" \
  "reason=$reason" \
  "dirty_reconstruction=HEAD+tracked.patch+untracked.tar.gz" \
  "interpretation=Only ALLOWED permits wording that the run used the recorded exact commits." \
  > "$OUT_DIR/VERDICT.txt"

echo "[provenance] $RUN_LABEL -> $OUT_DIR (exact_commit_claim=$claim)"
