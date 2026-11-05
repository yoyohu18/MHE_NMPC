# magnetic_gripper — 接近触发的磁吸式夹爪 (gz-sim 8 / Harmonic)

无人机飞到目标物体正上方一定范围内、相对速度近零时,把物体用一个 **fixed
`DetachableJoint`** 刚性焊到机体上,带着一起飞;不需要时 detach 让物体自由下
落。joint 只当夹爪代理,**故意不模拟手指接触动力学** —— 研究关注的是飞控对
载荷质量/CoM/惯量突变的响应,不是抓取力学。

组成:
- `MagneticGripper.{hh,cc}` + `CMakeLists.txt` —— gz-sim 8 **world 级** system 插件。
- `../../offboard_test_acados/gripper/proximity_gripper_node.py` —— ROS 2 接近触发节点(**gz-transport 直读位姿、直发 attach/detach,不经 bridge**)。
- `../../offboard_test_acados/gripper/gripper_flight_node.py` —— 简易 MAVROS 位置控制飞行节点(takeoff→飞到 box→吸附→带载悬停→投放),用于纯夹爪演示。
- `../../config/gripper/gripper_params.yaml` —— 阈值/目标参数。
- `../../config/gripper/gripper_bridge.yaml` —— ros_gz_bridge 配置(**现已不需要**,保留作参考;见 §4.2)。
- `../../worlds/gripper/gripper_test.sdf` —— 含 x500 spawn 位 + 一个 box + 全套 PX4 system 插件的测试 world。
- `../../../scripts/gripper/run_sitl_gripper.sh` —— 一键跑"裸 MAVROS 控制 + 夹爪"完整演示。
- `../../../scripts/gripper/run_sitl_gripper_viz.sh` —— 一键跑当前主配置的
  "acados NMPC + MHE + 夹爪" GUI 实验。
- `../../../scripts/gripper/run_gripper_headless.sh` —— 同一飞行栈的无头/批量入口。

> **先看结论**:100g 载荷下整套 pick→carry→hover→drop 完全跑通、飞行稳定;
> **0.3kg 及以上,acados NMPC + MHE 也压不住会发散**——原因不是质量,是吊挂的
> 未建模摆动/CoM 动力学,MHE 估质量标量补不了。完整实验与结论见 **§8**。

---

## 1. 为什么用自定义插件,而不是 gz-sim 自带的 DetachableJoint system

我读了已安装的 `libgz-sim8-detachable-joint-system.so` 里的实际符号(不是看
文档):gz-sim 8 的自带 system **比参考插件那个老版本强**,它支持
`<attach_topic>` / `<detach_topic>` / `<output_topic>`,内部有 `Already attached`
/ `Already detached` 守卫,**能 detach 后重复 attach**。所以"无法重复 attach"
这条对它**不成立**。

但它对本任务有两条硬伤:

1. **目标写死在 SDF**:`<child_model>` / `<child_link>` 是必填参数,加载时就
   固定。无法在运行时从"目标物体名列表"里动态选一个去吸。
2. **它是 model 级插件,parent 永远是宿主模型**:一个实例只能服务一对
   (parent, child)。表达不了"一个夹爪按需换目标"。

还有一条更重要的、来自旧实验栈的历史教训:自带
system 在 **Configure(加载即)** 阶段就把 child 焊上去。**一个从 t=0 就存在
的 fixed joint,会把无人机焊到一个搁在地面/远处的物体上**,起飞时控制器去对
抗这个"把机体拽向接地点"的约束 → 进入 ~5.6s 周期、roll/pitch 20–37°、推力饱
和、**永不恢复**的振荡(本栈用自带方案反复踩到,哪怕把 payload 挪到 50m 外或
已静止也照炸)。

**本插件做成 world 级,且关节只在收到 attach 命令时才创建 ——绝不在加载时存
在。** 配合接近节点(只在无人机正悬停在目标上方、相对速度近零时才发 attach),
fixed 约束锁的是一个很小的相对位姿、且无速度失配 → 没有冲击、没有对抗约束。
这正是规避上述历史振荡的关键。

> 一句话:自带的只在"永远只吸那一个已知 box、且能接受加载即焊接"时才够用 ——
> 本任务两条都不满足,所以走自定义 world 插件。

---

## 2. 构建

```bash
cd src/offboard_test_acados/gz_plugins/magnetic_gripper
cmake -B build -S .
cmake --build build      # 产出 build/libmagnetic_gripper.so
```

依赖:gz-sim8 / gz-plugin2 / gz-transport13 / gz-msgs10 / gz-math7(均随
Harmonic 安装)。

让 gz 找到插件:

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
```

ROS 2 节点随包构建:

```bash
cd ~/ros2_ws_HJH
colcon build --packages-select offboard_test_acados
source install/setup.bash
```

---

## 3. 消息接口

| 话题 | 类型 (gz / ros) | 方向 | 载荷 |
|------|-----------------|------|------|
| `/gripper/attach` | `StringMsg` / `std_msgs/String` | →插件 | `parentModel parentLink childModel childLink` |
| `/gripper/detach` | `StringMsg` / `std_msgs/String` | →插件 | 同上 |
| `/gripper/state`  | `StringMsg` / `std_msgs/String` | 插件→ | `ATTACHED/DETACHED <key> sim_time=<s> entity=<id>` |

载荷是 4 个空格分隔的 token(parent model/link、child model/link)。状态行带
**仿真时间戳**,用来给后续 estimator-reset 实验取 attach/detach 的精确时刻。

SDF 参数(`<plugin>` 块内,均可选,缺省即上表默认名):
`<attach_topic>`、`<detach_topic>`、`<state_topic>`。

---

## 4. 运行

### 4.1 world SDF 接线(已在 `worlds/gripper/gripper_test.sdf` 内)

```xml
<world name="gripper_test">
  <plugin filename="magnetic_gripper"
          name="magnetic_gripper::MagneticGripper">
    <attach_topic>/gripper/attach</attach_topic>
    <detach_topic>/gripper/detach</detach_topic>
    <state_topic>/gripper/state</state_topic>
  </plugin>
  <!-- ground_plane + box(独立 model,仅靠 contact 搁地面)+ light ... -->
</world>
```

### 4.2 位姿来源:gz-transport 直连(不用 bridge)

接近节点**直接用 `gz.transport13`** 订阅 `/world/<world>/pose/info`
(`gz.msgs.Pose_V`)读无人机和 box 的位姿、并直接 `advertise` 把
attach/detach 的 `gz.msgs.StringMsg` 发给插件。**完全不需要 ros_gz_bridge。**

> **为什么放弃 bridge**(踩过的坑):ros_gz_bridge 把 `Pose_V` 转
> `tf2_msgs/TFMessage` 时,**每个 transform 的 `child_frame_id` 会丢成空串**,
> 接近节点因此认不出哪个是无人机、哪个是 box。gz-transport 直读的 `Pose_V`
> 每个 `Pose` 自带 `name`(顶层 model 即 model 名),可靠。`gripper_bridge.yaml`
> 保留仅作参考,当前流程用不到。

### 4.3 接近触发节点

```bash
ros2 run offboard_test_acados proximity_gripper_node \
    --ros-args --params-file .../config/gripper/gripper_params.yaml -p drone_model:=x500_0
```

参数(`config/gripper/gripper_params.yaml`):

| 参数 | 含义 | 默认 |
|------|------|------|
| `world_name` | gz world 名(定 pose 话题) | `gripper_test` |
| `drone_model` / `parent_link` | 无人机 model 名 / 焊接到的机体 link | `x500_0` / `base_link` |
| `targets` / `child_link` | 候选目标 model 名列表 / 目标 link | `[box]` / `box_link` |
| `r_xy` | 水平距离上限 [m] | `0.15` |
| `h_min` / `h_max` | 相对高度区间 (无人机.z−目标.z) [m] | `0.3` / `1.5` |
| `v_rel_max` | 相对速度模长上限 [m/s] | `0.2` |

触发 attach 需**全部满足**:`d_xy < r_xy` 且 `dz ∈ [h_min,h_max]` 且
`|v_rel| < v_rel_max` 且 `/gripper/enable = true`。
detach:`/gripper/release = true`,或 `/gripper/enable` 拉低。

> `r_xy` 默认收紧到 **0.15m**(而非最初的 0.5):载荷焊接点在机体正下方,
> `r_xy` 越大、box 越可能焊在**侧下方**,产生大水平力臂、吸附瞬间就把姿态掀翻。
> 0.15 保证 box 几乎正下方才吸,力臂最小。

---

## 5. 正确性约束

- **attach 必须在相对速度近零时发生**(`v_rel_max` 可调)。瞬间施加的刚性
  fixed 约束会把两个 link 的速度强行拉到一致;若 attach 瞬间二者速度差很大,
  这个"瞬时速度跳变"就是一个冲量(理论上无穷大加速度的 jerk),轻则把姿态打
  飞、重则让求解器发散把仿真打崩。所以条件里强制相对速度 < 阈值。
- **detach 是瞬间删关节、载荷自由下落**;无人机突然失重上冲,是**预期物理现
  象**,不要去"平滑"它 —— 那正是要研究的载荷突变响应。
- **焊接几何**:fixed joint 锁的是 attach **瞬间**两 link 的相对位姿,box 不
  瞬移。接近节点保证 box 在机体正下方 r_xy 内、高度差 h_min..h_max,所以锁住
  的偏置小而合理;这个偏置带来的力臂力矩正是 CoM/惯量突变的来源。
- **box 不固定到地面**,只靠 contact 搁着 → attach 后推力超过合重即可把它拉
  离地面(单边接触自然分离)。⚠️ 但载荷偏重时,"被焊到还没离地的 box 上"这段
  等于被拴在地面,是重载发散的诱因之一——见 §8。所以重载要**吸附后尽快抬升离地**
  缩短拴系时间(即便如此 0.3kg 仍压不住,根因是摆动/CoM,见 §8.2)。
- **不要 self-reference**:DetachableJoint 连同一个 model 内两 link 是**静默
  no-op**(本栈踩过)。插件已显式拒绝 parent==child 的请求。box 是独立 model,
  天然满足跨模型要求。
- **物理引擎 = dartsim**(Harmonic 默认,`gripper_test.sdf` 里显式 `type="dart"`)。
  fixed DetachableJoint 在 dartsim 下稳定。

---

## 6. 验收(端到端,可复现)

> 前提:PX4 SITL 把 plain x500 spawn 进 `gripper_test` world。把
> `worlds/gripper/gripper_test.sdf` 放到 PX4 的 `Tools/simulation/gz/worlds/`(或软链),
> 并设 `GZ_SIM_SYSTEM_PLUGIN_PATH` 指向插件 build 目录,然后:
> ```bash
> export PX4_GZ_WORLD=gripper_test
> export PX4_SYS_AUTOSTART=4001          # 4001 = gz_x500(空机)
> make px4_sitl gz_x500
> ```
> 起 bridge 和接近节点(见 §4)。

1. **确认插件就绪 / 话题在线**
   ```bash
   gz topic -l | grep gripper          # /gripper/attach /detach /state
   ```
2. **无人机起飞、平移到 box (1,0,0.075) 正上方悬停**(用你的 offboard 控制器
   或 QGC 飞过去,悬停在 ~(1,0,1))。
3. **使能夹取**
   ```bash
   ros2 topic pub --once /gripper/enable std_msgs/Bool "{data: true}"
   ```
   条件满足后接近节点自动发 attach。
4. **确认 attach + 关节实体被创建**
   ```bash
   ros2 topic echo /gripper/state        # 看到 ATTACHED ... sim_time=...
   gz topic -e -t /gripper/state -n 1    # 同样,直接从 gz 侧看
   ```
   确认 box 跟随机体:平移/拉高无人机,观察 box 一起动
   ```bash
   gz topic -e -t /world/gripper_test/pose/info -n 1 | grep -A4 '"box"'
   ```
5. **飞一段后释放**
   ```bash
   ros2 topic pub --once /gripper/release std_msgs/Bool "{data: true}"
   ros2 topic echo /gripper/state        # DETACHED ... sim_time=...
   ```
   确认 box 自由下落(z 单调减小直到落地),无人机瞬间上冲 —— 预期物理。

直接用 gz topic 手动测插件(不经 ROS,验证插件本身):
```bash
# attach: 把 box::box_link 焊到 x500_0::base_link
gz topic -t /gripper/attach -m gz.msgs.StringMsg \
    -p 'data:"x500_0 base_link box box_link"'
# detach
gz topic -t /gripper/detach -m gz.msgs.StringMsg \
    -p 'data:"x500_0 base_link box box_link"'
```

---

## 7. 一键演示脚本

上述脚本都自带**启动前清理**(杀掉上一次残留的 gz/px4/节点 ——
不清会踩两个坑:① px4-rc.gzsim 检测到"已有 world 在跑"直接接旧 gz 服务器,新
PX4 的 EKF 和实际 gz 无人机脱节;② 残留的旧飞行/接近节点和新节点抢同一架无人机
的 setpoint / 反复 enable-disable 把 box churn 得吸了又松)。

### 7.1 `run_sitl_gripper.sh` —— 裸 MAVROS 控制 + 夹爪(演示用,稳)

空机 `gz_x500` + `gripper_flight_node`(纯位置 setpoint):
takeoff → 飞到 box 上方 → 接近节点吸附 → 平滑抬升带载悬停 → DROP 投放。
**100g 载荷下全程稳定、完整跑通**(见 §8)。GUI 可见,QGC 新开(解锁要 GCS 心跳)。

`gripper_flight_node` 关键点:
- setpoint 必须给**合法四元数**(`orientation.w=1`);留默认 (0,0,0,0) 会让 MAVROS
  解出 NaN yaw,PX4 位置控制发散飞走。
- DROP 阶段发 `enable=False`(不是发一次 `/gripper/release`)——飞行节点持续发
  `enable=True` 会把单次 release 顶掉、box 立刻被重抓。

### 7.2 `run_sitl_gripper_viz.sh` / `run_gripper_headless.sh` —— acados NMPC + MHE + 夹爪

空机 `gz_x500` 由 `acados_nmpc_node`(姿态+推力控制)飞,`gripper_mode:=true`:
空机起飞 → 低空悬停到 box 上方 → 吸附 → 定时抬升离地 → MHE 用电机转速反算真实
推力在线估质量、闭环喂回 NMPC。`acados_nmpc_node` 新增的相关参数(默认全关,
`gripper_mode=False` 时行为与基线逐字节一致):

| 参数 | 含义 |
|------|------|
| `gripper_mode` | 开夹爪模式(hover 固定悬停 + 空机初始 m_est + 关 mass_changer drop) |
| `grip_x/grip_y` | 悬停/box 的水平位置 |
| `grip_z_low` → `grip_z_high` | 吸附时低空悬停高度 → 抬升后高度 |
| `grip_lift_after_sec` / `grip_lift_dur` | NMPC 接管后多久开始抬升 / 抬升时长 |
| `use_mhe` | False 时 MHE 只诊断不喂回(m_est 固定),用于隔离验证控制器 |
| `grip_payload_mass` / `grip_mass_step_sec` | 诊断:定时把 m_est 手动阶跃到带载真值(绕过 MHE) |

MHE 节点用 `motor_speed_topic:=/x500_0/command/motor_speed`(空机的电机话题;
基线 x500_payload 是另一个,已做成参数)。

---

## 8. acados NMPC + MHE 重载实验与结论(关键)

目标:验证"用 MHE 在线估质量、闭环喂回 acados NMPC"能否扛住裸控制器扛不住的
DetachableJoint 吊挂载荷。**结论:不能,而且原因比"质量"更底层。**

### 8.1 实测结果

| 载荷 | 控制器 | 结果 |
|------|--------|------|
| **0.1kg** | 裸 MAVROS 位置控制 | ✅ **稳**。完整 takeoff→吸附→带载悬停(机-箱 Δz 恒定、box 离地)→投放坠落,全程 pos 抖动 <7cm。 |
| **0.3kg** | acados NMPC + MHE 闭环 | ❌ 发散。MHE 质量估计在边界 1.0↔5.0 乱跳,垃圾 m_est 灌进 NMPC → 崩。 |
| **0.3kg** | acados,MHE 反馈**关**(m_est 固定空机 2.06) | ❌ 发散(t≈8s)。 |
| **0.3kg** | acados,**手动把 m_est 阶跃到正确带载 2.36** + 吸附后立刻抬升去掉地面拴系 | ❌ **仍发散**(抬升成摆动吊挂那一刻,t≈8–13s,pos_err→10m)。 |

### 8.2 为什么 MHE 补不了 —— 缺的物理不是质量

关键的第 4 行实验排除了所有"质量"因素:**正确质量也直接喂了、地面拴系也去了,
照样发散**。所以:

- DetachableJoint 吊挂的 box **不是"质量阶跃",而是一套未建模动力学**:
  1. **CoM 偏移**——载荷焊在机体下方 0.4~0.6m,合成质心大幅下移,推力对质心有力臂;
  2. **摆动(pendulum)**——box 刚性连接但整体是个倒摆,机体一动 box 就甩;
  3. 抬升离地的过渡冲击 + box 接地时的"焊到地面"拴系。
- acados 的 NMPC 模型是**刚体四旋翼,只有一个质量标量 `m`**。它的预测里根本没有
  "下面挂着会甩的东西",实际与预测越差越远 → SQP 的 `res_stat` 从 1e-7 飙到 1e3、
  `sqp_iter` 顶满 100 → 求解失败 → 发散。
- **MHE 估的是质量标量**;就算估得完美(第 4 行直接喂了完美质量),也补不了 CoM
  偏移和摆动。**缺的物理是吊挂动力学本身,不是质量。**
- 100g 能飞,是因为吊挂扰动小到刚体模型的反馈能压住(5% 机重);0.3kg(15%)就超了。

> 这也解释并印证了本栈当初为什么放弃 DetachableJoint、改用 `mass_changer`
> (直接改写 base_link 惯量:是**纯质量/惯量变化、无独立刚体、无摆动**)——对
> "在线质量估计能否收敛"这个问题,mass_changer 是稳定可控的正确抽象;要研究
> **真吊挂**,得改控制/模型(下节)。

### 8.3 要真扛住吊挂,下一步方向(超出"估质量")

1. **把吊挂建进 NMPC**:状态里加载荷相对位姿(倒摆模型),或至少把 CoM 偏移建进
   动力学,让 NMPC "看得见"摆动去主动补偿(要动 `acados_model.py`)。
2. **估 CoM 偏移量**(不只质量),前馈补偿力臂力矩。
3. 折中摸底:大幅放慢抬升 + 更轻载荷,找"裸 acados 能扛的载荷上限"。

---

## 9. 集成踩坑记录(PX4 SITL × gz × 夹爪)

按遇到顺序,均已在代码/脚本里修掉:

1. **world 内联 `<plugin>` 顶掉 PX4 默认 system**:gz-sim 里 world SDF 一旦写了任何
   `<plugin>`,就不再自动加载 PX4 `server.config` 的默认 system。结果没有
   SceneBroadcaster(无 `scene/info` 服务)→ PX4 卡死在 "Waiting for Gazebo world"、
   无人机永远 spawn 不出来;也没有 Imu/Mag/Baro/NavSat → EKF 没数据。**修**:把
   整套 PX4 system 插件显式抄进 `gripper_test.sdf`(和 magnetic_gripper 并列)。
2. **解锁被拒 "Resolve system health failures first"**:缺 GCS 心跳。**修**:脚本
   每次新开一个 QGroundControl 连本次 PX4 实例。
3. **ros_gz_bridge 丢实体名**:`Pose_V`→`TFMessage` 的 `child_frame_id` 变空串。
   **修**:接近节点改 gz-transport 直读(见 §4.2)。
4. **飞行节点零四元数**:PoseStamped 默认 `orientation=(0,0,0,0)` → MAVROS 解出
   NaN yaw → 发散飞走。**修**:置单位四元数 `w=1`。
5. **残留进程串扰**(反复踩):旧 gz 世界没杀干净被新 PX4 误接;两个飞行/接近节点
   抢舵、反复 enable/disable churn。**修**:脚本启动前彻底清理并**验证 0 残留**。
6. **吸附力臂**:`r_xy` 太松 → box 焊在侧下方大力臂掀翻。**修**:`r_xy`→0.15。
7. **投放被重抓**:飞行节点持续发 `enable=True` 顶掉单次 release。**修**:DROP 阶段
   改发 `enable=False`。

---

## 10. 移植说明(相对参考 AttachablePlugin)

参考 https://github.com/Dagu12s/AttachablePlugin 的思路(运行时建/删
DetachableJoint 组件、topic 收发 StringMsg),但**重写**而非照搬:

- 命名空间 `ignition::gazebo` → `gz::sim`;头文件 `gz/sim/...`、
  `gz/transport/...`、`gz/msgs/...`;宏 `GZ_ADD_PLUGIN`/`GZ_PROFILE`;日志
  `gzmsg/gzwarn/gzerr`。
- 参考里的 detach 循环 `for(i=0;i<=list.size();i++)` 是**越界 bug**。本实现
  不用按索引遍历的删除写法,改成按 `map<key,Entity>` 直接定位 + 空安全;逻辑
  上等价于"`< size()` 且空表保护"。
- 做成 **world 级**(参考是 model 级),以支持运行时任意目标 + 不在加载时建关节
  (见 §1)。
- 所有 ECM 写操作(CreateEntity / CreateComponent / RequestRemoveEntity)都
  在 `PreUpdate`(仿真线程)做;transport 回调只解析入队,不碰 ECM。
