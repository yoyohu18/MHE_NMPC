#!/usr/bin/env python3
"""接近触发夹爪节点 (ROS 2 + gz-transport)。

监听无人机和目标 box 的位姿,当**全部**条件满足时,往夹爪插件发 attach
命令,把 box 焊到机体上:
  - 水平距离 < r_xy  ← 这一条同时是**偏心上限**,见 should_attach 里的 08-31 注释
  - 相对高度 (无人机.z - 目标.z) ∈ [h_min, h_max]  (无人机在目标上方)
  - 机体与目标的相对速度模长 < v_rel_max  (避免瞬间刚性约束 + 速度失配
    产生冲击 jerk 把仿真打崩)
  - 收到外部"夹取使能"信号 /gripper/enable (std_msgs/Bool = true)

detach:收到 /gripper/release (std_msgs/Bool = true),或 enable 拉低时释放。
`detach_delay_sec` 可仅在故障注入实验中延迟真正的 DetachableJoint 移除；
默认 0.0 s，因此不改变正常飞行或历史实验。

位姿来源 / attach 命令出口都走 **gz-transport 直连**,不经 ros_gz_bridge:
  - 订阅 gz 的 /world/<world>/pose/info(gz.msgs.Pose_V),里面每个 Pose 自带
    name(顶层 model 即 model 名),所以能可靠区分无人机和 box。
    (ros_gz_bridge 把 Pose_V 转 TFMessage 时 child_frame_id 会丢成空串,
     用不了 —— 这是当初踩的坑,所以这里直接读 gz。)
  - attach/detach 用 gz.msgs.StringMsg 直接 publish 到插件订阅的 topic。
无人机和 box 同在 gz ENU 世界系,速度用位姿有限差分得到。

只有"夹取使能/释放"这两个外部信号走 ROS(方便飞行节点/人工触发)。
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, Float64MultiArray

from ..lifecycle_trace import evt_line

import gz.transport13 as gztransport
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.stringmsg_pb2 import StringMsg


class TrackedPose:
    """一个实体的最近一帧位姿 + 估计速度。"""

    def __init__(self):
        self.t = None
        self.pos = None
        self.vel = (0.0, 0.0, 0.0)

    def update(self, t, pos):
        if self.t is not None and t > self.t:
            dt = t - self.t
            if dt > 1e-4:
                self.vel = tuple((pos[i] - self.pos[i]) / dt for i in range(3))
        self.t = t
        self.pos = pos


class ProximityGripperNode(Node):
    def __init__(self):
        super().__init__('proximity_gripper_node')

        self.declare_parameter('world_name', 'gripper_test')
        self.declare_parameter('drone_model', 'x500_0')
        self.declare_parameter('parent_link', 'base_link')
        self.declare_parameter('targets', ['box'])
        self.declare_parameter('child_link', 'box_link')
        # r_xy 有双重语义:①够不够得着 ②**允许的抓取偏心上限**。
        # 【2026-08-31】②才是真正吃紧的那一条。box 固定在 world 的 (1.0, 0.0),
        # 而 drone 悬停点是 grip_y=GRIP_ECC_Y —— 偏心是**故意制造的实验变量**,
        # 设计值就是 ECC_Y。但跟踪误差 + 旋翼下洗吹动 box 会把实际 d_xy 放大:
        # 266 次历史 attach 的分布是 中位 0.096 / p90 0.139 / max 0.200,
        # 而设计值只有 0.10。
        # 为什么必须卡住上限:带偏心悬停需要一个**常值配平 roll 力矩**
        #   tau_roll = c_y * T = (m_P/m_T) * r_y * T ≈ 2.944 * r_y   (0.3kg 载荷, T≈23.2N)
        # 而 NMPC 的 roll 约束是 ±0.5 N·m。r_y=0.17 时配平就吃掉 99%,figure8
        # **没有任何机动余量** → 姿态发散 → 倾角 35° → cos 损失 → 掉高坠机
        # (gviz_20260831_165224 实测:tau_phys roll 在 figure8 之前就贴到 −0.537)。
        # 408 轮历史回归(飞行≥60s):|r_y|≥0.15 坠机 5/13=38.5%,0.12~0.15 为
        # 15.5%,<0.12 为 9.0%(单调,Fisher p=0.0056)。
        # 门槛与配平占比的换算:0.12→71%  0.13→77%  0.15→88%  0.17→100%。
        # 默认 0.13 = ECC_Y 标称 0.10 + 0.03 容差,配平占 77%、留 23% 给机动;
        # 对应历史拒绝率 15.4%(0.15 只拒 4.9% 但放行了高危档,0.12 拒 23.7%)。
        self.declare_parameter('r_xy', 0.13)
        # 被拒时的节流日志间隔 [s]。0 = 不打(旧行为)。**默认要打**:收紧门槛后
        # "为什么迟迟不 attach"必须看得见,否则轮次挂在低空而日志里一片空白。
        self.declare_parameter('reject_log_period', 2.0)
        # enable 之后多久还没 attach 就告警 [s]。只 WARN 不改变行为——drone 仍在
        # 跟踪 (grip_x, grip_y),d_xy 通常会自己收敛回设计值;但若 box 被下洗吹偏
        # 就再也回不来,那时需要这条 WARN 才能判断是"还在等"还是"永远等不到"。
        self.declare_parameter('attach_timeout_warn', 8.0)
        self.declare_parameter('h_min', 0.3)
        self.declare_parameter('h_max', 1.5)
        self.declare_parameter('v_rel_max', 0.2)
        self.declare_parameter('attach_topic', '/gripper/attach')
        self.declare_parameter('detach_topic', '/gripper/detach')
        self.declare_parameter('enable_topic', '/gripper/enable')
        self.declare_parameter('release_topic', '/gripper/release')
        self.declare_parameter('detach_delay_sec', 0.0)
        # 永久卡死故障注入(2026-09-15,默认关):释放请求(enable=false 或 release)
        # 到达时关节保持吸附、永不脱开,t_sep=+inf。与 detach_delay_sec 互斥。
        self.declare_parameter('detach_jam', False)
        # 部分丢失故障注入(2026-09-15,默认空=关):"box4:45,box3:80,box2:115" 表示首次
        # ATTACH 后 45 s 松开 box4 ……;前缀 "dry:" 只打标记不松开(对照臂)。被松开的
        # 目标永久拉黑、不再重吸(否则 ~40 ms 就被吸回,见 on_release 注释)。
        # NMPC/MHE 不收任何信号 —— 这是非计划的部分载荷丢失。
        self.declare_parameter('partial_loss_schedule', '')
        # 吸附持续判据(2026-09-16,默认 0 = 旧行为:判据首次满足的那一 tick 就吸)。
        # 旧行为下偏心在 0.13~0.16 间来回摆时,"掠过"上限的瞬时值就触发吸附,吸附点
        # 堆在边界上(E1 冒烟 5/6 轮 d_xy∈[0.123,0.130]),而吸附后 1~4 s 力矩饱和
        # 振荡的坠机都出在这一档。>0 时三条判据须**连续**满足这么久才吸,任一 tick
        # 不满足计时清零。
        self.declare_parameter('attach_dwell_sec', 0.0)
        # 偏心下限(默认 0 = 不设)。偏心过小时一阶矩参考 s_ref 也小,释放判据
        # |s|<ratio·s_ref 的绝对门槛随之变低,机动噪声下迟迟压不进去(冒烟 C_SAME/d0:
        # d_xy=0.058, s_ref=0.0070, 脱离后 15.1 s 才清模)。设为 ECC_Y−TOL 可把偏心
        # 这个检测延迟协变量收在设计值附近。
        self.declare_parameter('attach_r_min', 0.0)

        g = self.get_parameter
        world = g('world_name').value
        self.drone_model = g('drone_model').value
        self.parent_link = g('parent_link').value
        self.targets = list(g('targets').value)
        self.child_link = g('child_link').value
        self.r_xy = float(g('r_xy').value)
        self.h_min = float(g('h_min').value)
        self.h_max = float(g('h_max').value)
        self.v_rel_max = float(g('v_rel_max').value)
        self.reject_log_period = float(g('reject_log_period').value)
        self.attach_timeout_warn = float(g('attach_timeout_warn').value)
        self.detach_delay_sec = max(
            0.0, float(g('detach_delay_sec').value))
        self.detach_jam = bool(g('detach_jam').value)
        if self.detach_jam and self.detach_delay_sec > 0.0:
            raise ValueError('detach_jam and detach_delay_sec>0 are mutually exclusive')
        self._jam_logged = set()
        self.attach_dwell_sec = max(0.0, float(g('attach_dwell_sec').value))
        self.attach_r_min = max(0.0, float(g('attach_r_min').value))
        if self.attach_r_min >= self.r_xy:
            raise ValueError('attach_r_min must be < r_xy')
        self._dwell_start = {}          # target -> 判据开始连续满足的单调钟
        sched = str(g('partial_loss_schedule').value).strip()
        self.partial_dry = sched.startswith('dry:')
        if self.partial_dry:
            sched = sched[4:]
        self.partial_schedule = []
        for item in filter(None, (x.strip() for x in sched.split(','))):
            tgt, t = item.split(':')
            self.partial_schedule.append((float(t), tgt.strip()))
        self.partial_schedule.sort()
        self._first_attach_mono = None
        self._blocked = set()
        self._last_reject = None        # 最近一次拒绝的原因串,或 None
        self._last_reject_log = 0.0     # 上次打拒绝日志的墙钟
        self._enable_stamp = None       # 收到 enable=true 的墙钟
        self._timeout_warned = False

        self.enabled = False
        self.poses = {self.drone_model: TrackedPose()}
        for t in self.targets:
            self.poses[t] = TrackedPose()
        self.attached = set()
        self._pending_detach = {}

        # gz-transport:订阅 pose/info,广告 attach/detach。
        self.gz = gztransport.Node()
        self.pose_topic = f'/world/{world}/pose/info'
        self.gz.subscribe(Pose_V, self.pose_topic, self.on_pose_gz)
        self.attach_pub = self.gz.advertise(
            g('attach_topic').value, StringMsg)
        self.detach_pub = self.gz.advertise(
            g('detach_topic').value, StringMsg)

        # ROS:attach 瞬间的真实几何偏移 [rx,ry,rz] = box位置 - 机体位置
        # (世界系;attach 发生在机体近水平悬停时,近似等于机体系)。NMPC 节点
        # 用它算复合质心偏移 c_xy 和吊挂惯量增量 dJ——两次实测 attach 的
        # dz=0.593/0.393、d_xy=0.041/0.109 差异都很大,写死的 grip_arm_d
        # 参数只配当兜底。TRANSIENT_LOCAL 让晚启动/重启的订阅方也能拿到。
        self.offset_pub = self.create_publisher(
            Float64MultiArray, '/gripper/attach_offset',
            QoSProfile(depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # ROS:全部载荷真实脱开的时刻。只供 release_baseline=ORACLE(性能上界)
        # 订阅;主线 MHE/NMPC 不订阅,不构成外部事件信号。
        self.sep_truth_pub = self.create_publisher(Empty, '/gripper/sep_truth', 10)
        # ROS:吸附前"目标 - 机体"相对位置 [dx,dy,dz](世界系轴向),相当于理想
        # 下视相机。只供 NMPC 的最后对准(grip_align_rel)用,MHE 不订阅。
        # 2026-10-29:PX4 EKF 本地系相对 gz 世界系每轮有 2~4cm 水平漂移,按名义
        # 坐标悬停时 d_xy 落在 0.011~0.043,r_xy=0.03 下 8 轮里 3 轮没当场吸上。
        self.target_rel_pub = self.create_publisher(
            Float64MultiArray, '/gripper/target_rel', 1)

        # ROS:只收使能/释放。
        self.create_subscription(Bool, g('enable_topic').value,
                                 self.on_enable, 10)
        self.create_subscription(Bool, g('release_topic').value,
                                 self.on_release, 10)
        self.create_timer(0.1, self.tick)

        self.get_logger().info(
            f'proximity_gripper ready (gz-transport): drone={self.drone_model} '
            f'targets={self.targets} pose_topic={self.pose_topic} '
            f'r_xy={self.r_xy} h=[{self.h_min},{self.h_max}] '
            f'v_rel_max={self.v_rel_max} '
            f'attach_dwell={self.attach_dwell_sec:.2f}s r_min={self.attach_r_min} '
            f'detach_delay={self.detach_delay_sec:.2f}s jam={self.detach_jam} '
            f'partial_loss={"dry:" if self.partial_dry else ""}{self.partial_schedule}')

    # --- gz-transport 回调(gz 线程)---
    def on_pose_gz(self, msg: Pose_V):
        for p in msg.pose:
            if p.name in self.poses:
                t = p.header.stamp.sec + p.header.stamp.nsec * 1e-9
                pos = (p.position.x, p.position.y, p.position.z)
                self.poses[p.name].update(t, pos)

    # --- ROS 回调 ---
    def on_enable(self, msg: Bool):
        if msg.data and not self.enabled:
            self.get_logger().info('gripper ENABLED')
        elif not msg.data and self.enabled:
            self.get_logger().info('gripper DISABLED -> releasing all')
        if msg.data and self._enable_stamp is None:
            self._enable_stamp = time.time()
            self._timeout_warned = False
        self.enabled = msg.data
        if not self.enabled:
            self._enable_stamp = None
            for tgt in list(self.attached):
                self.schedule_detach(tgt, 'enable=false')

    def on_release(self, msg: Bool):
        if msg.data:
            # 2026-09-15:释放同时关闭重吸。原实现只 detach 不动 enabled,tick() 在
            # box 刚脱开(仍与机体同速、在吸附范围内)时 ~40-80ms 就把它重新吸上
            # (08-30 与 09-15 两次实测),推力无阶跃 = 注入的意外脱落根本没发生。
            # 该话题全仓无其他发布方(历史批次均改用 enable=false 规避),口径不受影响;
            # 与 enable=false 的区别只在于 NMPC/MHE 订阅不到它(意外脱落注入要的正是这点)。
            self.enabled = False
            for tgt in list(self.attached):
                self.schedule_detach(tgt, 'release=true')

    # --- 主判定 ---
    def tick(self):
        self.flush_pending_detach()
        self._run_partial_schedule()
        self._publish_target_rel()
        if not self.enabled:
            self._dwell_start.clear()   # 重新使能后持续计时必须从头算
            return
        drone = self.poses.get(self.drone_model)
        if drone is None or drone.pos is None:
            return
        for tgt in self.targets:
            if tgt in self.attached or tgt in self._blocked:
                continue
            tp = self.poses.get(tgt)
            if tp is None or tp.pos is None:
                continue
            if self.should_attach(drone, tp):
                now = time.monotonic()
                t0 = self._dwell_start.setdefault(tgt, now)
                held = now - t0
                if held + 1e-9 < self.attach_dwell_sec:
                    self._last_reject = (f'判据已满足,持续 {held:.1f}/'
                                         f'{self.attach_dwell_sec:.1f}s')
                    self._log_reject(tgt)
                    continue
                self._dwell_start.pop(tgt, None)
                self.get_logger().info(
                    f'attach condition met: {self._met_desc} '
                    f'held={held:.2f}s (dwell {self.attach_dwell_sec:.2f}s)')
                self.send_attach(tgt)
                self.publish_offset(drone, tp)
            else:
                self._dwell_start.pop(tgt, None)
                self._log_reject(tgt)

    def _run_partial_schedule(self):
        if not self.partial_schedule or self._first_attach_mono is None:
            return
        el = time.monotonic() - self._first_attach_mono
        while self.partial_schedule and el >= self.partial_schedule[0][0]:
            t, tgt = self.partial_schedule.pop(0)
            if self.partial_dry:
                self.get_logger().warn(
                    f'PARTIAL LOSS (dry) "{tgt}" at +{el:.2f}s after first attach '
                    f'(marker only, still attached={tgt in self.attached})')
                continue
            self._blocked.add(tgt)
            was = tgt in self.attached
            if was:
                self.send_detach(tgt)
            self.get_logger().warn(
                f'PARTIAL LOSS injected "{tgt}" at +{el:.2f}s after first attach '
                f'(was_attached={was}, remaining={sorted(self.attached)})')

    def schedule_detach(self, tgt, source):
        """Schedule physical detach while preserving the command timestamp.

        The nonzero-delay path is a deliberate fault injector: the controller has
        issued its release command, but the simulated gripper remains physically
        attached until the deadline. Repeated commands do not extend the delay.
        """
        if tgt not in self.attached or tgt in self._pending_detach:
            return
        if self.detach_jam:
            if tgt not in self._jam_logged:
                self._jam_logged.add(tgt)
                self.get_logger().warn(
                    f'DETACH JAM injected for "{tgt}": source={source}, '
                    'physical joint held permanently')
                self.get_logger().info(evt_line('inject_jam', target=tgt, source=source))
            return
        if self.detach_delay_sec <= 0.0:
            self.send_detach(tgt)
            return
        deadline = time.monotonic() + self.detach_delay_sec
        self._pending_detach[tgt] = deadline
        self.get_logger().warn(
            f'DELAYED DETACH injected for "{tgt}": source={source}, '
            f'physical joint held for {self.detach_delay_sec:.2f}s')

    def flush_pending_detach(self):
        now = time.monotonic()
        for tgt, deadline in list(self._pending_detach.items()):
            if now >= deadline:
                self.get_logger().warn(
                    f'DELAYED DETACH elapsed for "{tgt}": issuing physical detach')
                self.send_detach(tgt)

    def should_attach(self, drone, tgt):
        """三条判据全过才 attach。不过时把**原因**写进 self._last_reject,
        由 tick 节流打印 —— 静默拒绝会让"迟迟不 attach"完全无法诊断
        (2026-08-31 收紧 r_xy 时补,见该参数声明处的偏心-配平力矩推导)。"""
        dx = drone.pos[0] - tgt.pos[0]
        dy = drone.pos[1] - tgt.pos[1]
        dz = drone.pos[2] - tgt.pos[2]
        d_xy = math.hypot(dx, dy)
        v_rel = math.sqrt(sum(
            (drone.vel[i] - tgt.vel[i]) ** 2 for i in range(3)))
        why = []
        if d_xy >= self.r_xy:
            # 偏心超限是**本节点最该讲清楚**的一条:顺带把它折算成配平力矩占比,
            # 免得看日志的人还要自己换算才知道"超了 0.02m 到底要不要紧"。
            why.append(f'偏心 d_xy={d_xy:.3f} ≥ r_xy={self.r_xy:.3f} '
                       f'(配平 roll ≈ {2.944 * d_xy:.3f}N·m = '
                       f'{100.0 * 2.944 * d_xy / 0.5:.0f}% 的 ±0.5 约束)')
        elif d_xy < self.attach_r_min:
            why.append(f'偏心 d_xy={d_xy:.3f} < r_min={self.attach_r_min:.3f}')
        if not (self.h_min <= dz <= self.h_max):
            why.append(f'高度 dz={dz:.3f} ∉ [{self.h_min},{self.h_max}]')
        if v_rel >= self.v_rel_max:
            why.append(f'相对速度 v_rel={v_rel:.3f} ≥ {self.v_rel_max}')
        if why:
            self._last_reject = '; '.join(why)
            return False
        self._last_reject = None
        # 日志留给 tick 在真正吸附时打(持续判据下本函数会连续多 tick 返回 True)。
        self._met_desc = (f'd_xy={d_xy:.3f} dz={dz:.3f} v_rel={v_rel:.3f} '
                          f'(r_xy 上限 {self.r_xy:.3f})')
        return True

    def _log_reject(self, name):
        """节流打印"为什么还没 attach",并在超时后升级成一条 WARN。"""
        if not self._last_reject:
            return
        now = time.time()
        if self.reject_log_period > 0.0 and \
                now - self._last_reject_log >= self.reject_log_period:
            self._last_reject_log = now
            self.get_logger().info(
                f'attach 暂不触发 [{name}]: {self._last_reject}')
        if (self._enable_stamp is not None and not self._timeout_warned
                and self.attach_timeout_warn > 0.0
                and now - self._enable_stamp >= self.attach_timeout_warn):
            self._timeout_warned = True
            self.get_logger().warn(
                f'enable 已 {self.attach_timeout_warn:.0f}s 仍未 attach [{name}]: '
                f'{self._last_reject} —— drone 仍在跟踪悬停点,d_xy 通常会自己收敛;'
                f'若 box 被下洗吹偏则不会,此时本轮不会进入 LIFT。')

    def _publish_target_rel(self):
        """吸附前发第一个未吸附目标的相对位置;全部吸上后停发(对准已无意义)。"""
        drone = self.poses.get(self.drone_model)
        if drone is None or drone.pos is None:
            return
        for tgt in self.targets:
            if tgt in self.attached or tgt in self._blocked:
                continue
            tp = self.poses.get(tgt)
            if tp is None or tp.pos is None:
                continue
            msg = Float64MultiArray()
            msg.data = [float(tp.pos[i] - drone.pos[i]) for i in range(3)]
            self.target_rel_pub.publish(msg)
            return

    def publish_offset(self, drone, tgt):
        msg = Float64MultiArray()
        msg.data = [float(tgt.pos[i] - drone.pos[i]) for i in range(3)]
        self.offset_pub.publish(msg)
        self.get_logger().info(
            f'-> attach offset (box - drone) = '
            f'[{msg.data[0]:+.3f}, {msg.data[1]:+.3f}, {msg.data[2]:+.3f}] m')
        self.get_logger().info(evt_line(
            'attach_offset', rx=msg.data[0], ry=msg.data[1], rz=msg.data[2]))

    # --- 发命令(gz-transport)---
    def send_attach(self, tgt):
        payload = (f'{self.drone_model} {self.parent_link} '
                   f'{tgt} {self.child_link}')
        msg = StringMsg()
        msg.data = payload
        self.attach_pub.publish(msg)
        self.attached.add(tgt)
        self.get_logger().info(f'-> ATTACH "{payload}"')
        self.get_logger().info(evt_line('attach', target=tgt, n_attached=len(self.attached)))
        if self._first_attach_mono is None:
            self._first_attach_mono = time.monotonic()

    def send_detach(self, tgt):
        payload = (f'{self.drone_model} {self.parent_link} '
                   f'{tgt} {self.child_link}')
        msg = StringMsg()
        msg.data = payload
        self.detach_pub.publish(msg)
        self._pending_detach.pop(tgt, None)
        self.attached.discard(tgt)
        self.get_logger().info(f'-> DETACH "{payload}"')
        self.get_logger().info(evt_line('sep', target=tgt, n_attached=len(self.attached)))
        if not self.attached:
            self.sep_truth_pub.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = ProximityGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
