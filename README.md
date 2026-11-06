# MHE_NMPC — 吊运四旋翼主线工作空间

从 `ros2_ws_HJH` 提炼出来的**主线**：速度+质量降阶 MHE（R）＋ 载荷自适应 NMPC ＋ 磁吸夹爪
＋ 加速度计力平衡释放确认（论文 arm D）。2026-11-05 建立。

旧工作空间里的全状态 14 维 MHE（F）、coupled / 一阶矩 / 遗忘因子 / 机动门控、
MHE 侧释放判据、各类消融开关和历史实验脚本都**不在**这里；需要复现旧实验时回到
`ros2_ws_HJH`。

## 目录

```
src/offboard_test_acados/            ROS2 包（包名沿用旧名，节点名与话题名不变）
  offboard_test_acados/
    mhe/                             降阶 MHE（本空间唯一的估计器）
      params.py                        全部参数的单一来源
      model.py                         x=[v_world, m]，p=[q_meas, T_phys]
      solver_builder.py                acados OCP（ERK4 / GN / SQP / HPIPM）
      weights.py                       事件降权时间表
      estimator.py                     窗口求解核心 + 质量健康检查（纯 Python）
      node.py                          ROS2 节点 mhe_node
    acados_nmpc_node.py ...          NMPC 节点（主线精简版，默认参数即主线取值）及其依赖
    gripper/proximity_gripper_node.py 夹爪节点
  test/                              单元测试（71 个）
  config/ worlds/ gz_plugins/        夹爪参数、Gazebo 世界、磁吸夹爪插件源码
scripts/
  run_sitl.sh                        一键启动主线 SITL（headless）
  run_batch.sh                       主线批次驱动（排程、代码指纹、逐架判定、补跑、汇总）
  stop_sitl.sh                       停止仿真
  tools/                             心跳、溯源记录、多箱世界生成、MHE/NMPC 新旧等价对拍工具
docs/reference_params_20261105/      录自旧空间运行中批次的节点实际参数（基准）
thesis/                              学位论文（R 为主线的版本，2026-11-05 自 ros2_ws_HJH/src/paper/thesis 复制）
  main.tex                             正文，pdflatex → bibtex → pdflatex ×2，150 页
  export/bilingual/ch3-4_bilingual.tex 第 3–4 章中英对照精简版，xelatex ×2
```

## 用法

```bash
cd /home/clear/MHE_NMPC
source /opt/ros/jazzy/setup.bash && colcon build && source install/setup.bash

# 单元测试
cd src/offboard_test_acados
ACADOS_SOURCE_DIR=/home/clear/acados LD_LIBRARY_PATH=/home/clear/acados/lib python3 -m pytest -q test/

# 一次完整任务（0.30 kg、2 m/s、立即脱离）
bash scripts/run_sitl.sh
PAYLOAD_KG=0.15 SPEED=4 DETACH_DELAY_SEC=4 bash scripts/run_sitl.sh   # 其他工况
bash scripts/stop_sitl.sh

# 批次(随机化排程 + 逐架判定 + 汇总,结果在 results/batch_<EXP>_<时间>/summary.md)
DRY_RUN=1 EXP=faults bash scripts/run_batch.sh       # 只看排程
EXP=faults bash scripts/run_batch.sh                 # smoke|release|faults|thrust|thrust_small|thrust_delay
BATCH_DIR=results/batch_faults_<时间> bash scripts/run_batch.sh   # 续跑
```

日志写到 `results/`。`run_sitl.sh` 的实验开关只有：`PAYLOAD_KG`、`SPEED`（2/4）、
`DETACH_DELAY_SEC`、`THRUST_FAULT_GAIN`、`UNCOMMANDED_LOSS`、`DETACH_JAM`、`RESID_LOG_DIR`；
MHE/夹爪的其余主线参数写死在脚本里；NMPC 的主线参数就是节点默认值，脚本只传
8 字速度（`grip_dyn_w`/`grip_dyn_ramp`）、`thrust_map_fault_gain`、`grip_uncommanded_loss`。
取值与旧空间主线逐项一致。

⚠️ 两个工作空间共用 `/home/clear/PX4-Autopilot`，进程名也相同。`run_sitl.sh` /
`stop_sitl.sh` 在批次锁 `/tmp/bodyrate_repeat.lock` 被占用或已有仿真进程时会拒绝执行，
**不要**绕过这个检查同时跑两个空间的仿真。

## 与旧空间的等价性

`scripts/tools/replay_mhe_node.py` 把一次飞行录下的 odometry + 电机转速流离线喂给节点，
`compare_mhe_frames.py` 逐帧比对发布出去的载荷帧。用旧空间 2026-11-05 批次的 5 架 R 飞行
（含推力 −10%、+10%、4 m/s、延迟脱离）比对旧 `mhe_node`（`MHE_MODEL_VARIANT=reduced`）
与新 `mhe/node.py`：NMPC 消费的字段（质量、dJ、J、空载置信度、健康位、解龄）以及质量话题
**逐帧完全相同**（差值 0），健康异常的触发次数一致。唯一差别是 `s_xy`/`c_xy` 两个字段，
新节点恒为 0；主线 NMPC 偏心补偿关闭，不消费它们。

NMPC 是闭环的，没有录好的输入流可回放，所以用 `scripts/tools/nmpc_equiv_harness.py`
离线对拍：把节点的发布/订阅/定时器/服务客户端/日志/时钟全部换成假的，用确定性合成
被控对象闭环驱动一整趟任务（接近→对准→吸附→抬升→8 字→投放→确认），记录全部发布
输出与运行期日志，再用 `nmpc_equiv_compare.py` 逐条比对。精简前后的节点在 7 个场景
（正常、延迟脱离 4 s、意外脱落、夹爪卡死→UNRESOLVED、推力增益 0.9、4 m/s、0.15 kg）下
**全部发布输出（19556 条）与运行期日志（5597 条）逐位相同**；新节点只传实验开关
（`MINIMAL=1`，其余全靠默认值）时同样逐位相同。

## 论文

`thesis/` 是今后的**主版本**：96 架 F/R 闭环结果、摘要和第 1/6 章的 draftnote 都在这里补。
`ros2_ws_HJH/src/paper/thesis` 原样保留在旧仓库（受 git 跟踪，未删除），不再同步。

## 现状与待办

- [x] 降阶 MHE 节点精简重写（旧 `mhe_node.py` 3371 行 → 新 `mhe/node.py` 583 行，另有 estimator/model 等小模块），逐帧等价已验证。
- [x] 启动脚本精简（旧 683 行、数百个开关 → 179 行、7 个开关）。
- [x] **SITL 闭环冒烟**（2026-11-05，`results/*_20261105_202843.*`）：0.30 kg / 2 m/s / 立即脱离，
      抓取→抬升→8 字→t=101.2 s 投放→2.4 s 后 acc 确认释放（置信度 0.998）；NMPC 求解失败 0 次，
      MHE 全程 healthy、求解失败 0 次，载荷估计 ≈0.296 kg（真值 0.30）。
      顺带修了 `run_sitl.sh` 在 `set -u` 下 source ROS setup 即退出的问题。
      注意：`stop_sitl.sh` 按命令行子串杀进程，调用它的 shell 命令行里若含 `gz sim`、
      `mavros_node` 等字样会被一起杀掉，请单独调用。
- [x] **NMPC 节点精简**（2026-11-05）：`acados_nmpc_node.py` 3712 行 → 1707 行。按主线参数
      常量折叠，删去旧接口/真值几何/online 几何、L1 与 fixed 控制基线、旧释放基线、MHE 侧
      释放确认、偏心补偿、c_xy 冻结、离地武装、LIFT hold、PX4 增益缩放、非夹爪模式、非解耦
      发布等分支；数值参数默认值改为主线取值，`run_sitl.sh` 只传实验开关。离线对拍 7 个场景
      逐位等价（见上）；SITL 回归（`results/*_20261105_211609.*`）：t=101.1 s 投放、2.4 s 后 acc
      确认释放，NMPC 求解失败 0 次，MHE 全程 healthy。单测删去只针对已移除功能的 33 个
      （原件仍在 `ros2_ws_HJH`），其余 78 个通过。`straight_reference.py` 已无引用，删除。
- [x] 死代码清理（2026-11-06）：删去 `acados_params.py` 里改变代码生成的环境变量开关
      （`NMPC_GEOM_COUPLED`、`NMPC_QUAT_ERR`、`NMPC_PAYLOAD_KI`、`NMPC_THRUST_BOX_LEGACY`）及
      coupled/euclid 分支、`payload_estimate.py` 的旧释放判据与乘积分数、MHE 的
      `no_payload_score_mode` 等。重新生成的 NMPC/MHE 求解器 C 代码与清理前逐字节相同，
      NMPC 离线对拍 4 个场景逐位等价。
- [ ] 论文：填入 96 架闭环配对结果（第 5 章 §5.7.3、摘要、第 1/6 章的 draftnote）。
- [ ] RViz 可视化启动脚本（旧空间的 `run_sitl_gripper_viz.sh`）尚未迁移。
