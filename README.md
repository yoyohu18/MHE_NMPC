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
    acados_nmpc_node.py ...          NMPC 及其依赖（目前与旧空间相同，见“现状”）
    gripper/proximity_gripper_node.py 夹爪节点
  test/                              单元测试（111 个）
  config/ worlds/ gz_plugins/        夹爪参数、Gazebo 世界、磁吸夹爪插件源码
scripts/
  run_sitl.sh                        一键启动主线 SITL（headless）
  stop_sitl.sh                       停止仿真
  tools/                             心跳、溯源记录、多箱世界生成、新旧等价回放工具
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
```

日志写到 `results/`。`run_sitl.sh` 的实验开关只有：`PAYLOAD_KG`、`SPEED`（2/4）、
`DETACH_DELAY_SEC`、`THRUST_FAULT_GAIN`、`UNCOMMANDED_LOSS`、`DETACH_JAM`、`RESID_LOG_DIR`；
其余主线参数写死在脚本里，取值与旧空间主线逐项一致。

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
- [ ] **NMPC 节点精简**：`acados_nmpc_node.py`（约 3700 行）目前原样拷贝，只改了一行
      import；里面仍有 L1、coupled、旧释放基线等非主线分支。删这些分支会改动控制器，
      必须逐步删、每步用 SITL 回归验证，计划在冒烟通过后进行。
- [ ] 论文：填入 96 架闭环配对结果（第 5 章 §5.7.3、摘要、第 1/6 章的 draftnote）。
- [ ] RViz 可视化启动脚本（旧空间的 `run_sitl_gripper_viz.sh`）尚未迁移。
