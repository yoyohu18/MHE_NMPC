#!/usr/bin/env python3
"""NMPC 节点离线对拍驱动:伪时钟 + 假发布/订阅/定时器/客户端 + 确定性合成被控对象。
用法: python3 nmpc_equiv_harness.py <node_file.py> <scenario> <out.pkl>
场景: nominal d4 loss jam k09 v4 p15;MINIMAL=1 只传实验开关(验证默认值即主线取值)。
对比: python3 nmpc_equiv_compare.py ref.pkl new.pkl [--logs]
旧版本作参考: git archive <rev> src/offboard_test_acados | tar -x -C /tmp/ref,
  再 PKG_SRC=/tmp/ref/src/offboard_test_acados python3 nmpc_equiv_harness.py \
  /tmp/ref/src/offboard_test_acados/offboard_test_acados/acados_nmpc_node.py nominal ref.pkl
同一场景下两个版本的节点收到逐位相同的输入序列(只要它们的输出逐位相同),
所以"全部发布输出逐位相同"即证明主线行为等价。
"""
import importlib.util
import math
import os
import pickle
import sys

import numpy as np

# PKG_SRC:被测节点所属的包目录(跑旧版本时指向从 git 导出的旧包,旧节点依赖旧模块)
SRC = os.environ.get('PKG_SRC') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'offboard_test_acados')
sys.path.insert(0, SRC)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from rosidl_runtime_py import message_to_ordereddict  # noqa: E402

node_file, scen, out_path = sys.argv[1:4]

# 精简前 run_sitl.sh 传给 NMPC 的全部主线参数(0.30 kg / 2 m/s)。驱动旧版节点
# (ros2_ws_HJH 的 acados_nmpc_node.py)当参考时用它;新版节点的默认值就是这组取值。
MAINLINE = dict(
    gripper_mode='true', grip_x='1.0', grip_y='0.0', grip_align_rel='true',
    grip_align_offset_x='0.0', grip_align_offset_y='0.0', grip_z_low='0.35',
    grip_z_high='6.0', grip_mass_step_sec='0.0', grip_payload_mass='0.3',
    grip_arm_d='0.27', grip_lift_after_sec='3.0', grip_lift_dur='8.4',
    grip_lift_hold_enable='false', grip_lift_hold_dz='0.35', grip_lift_hold_sec='3.0',
    dj_track_mest='true', drop_publish_mass_event='false', grip_mp_cap='0.6',
    grip_dj_floor_mp='0.05', dj_ratchet_enable='false',
    attach_j_bootstrap_hold_until_drop='true', use_mhe='true',
    continuous_payload_estimates='true', release_baseline='P',
    release_confirm_source='acc', acc_loss_watchdog_enable='true',
    acc_loss_arm_hold_sec='2.0', decouple_publish='true', publish_hz='50.0',
    geom_source='estimate', geom_release_mode='event', grip_payload_envelope='0.3',
    attach_j_bootstrap_enable='true', attach_j_bootstrap_on_liftoff='false',
    eccentricity_compensation='false', cxy_freeze_in_maneuver='false',
    cxy_freeze_window_sec='3.0', liftoff_detect_dz='0.03', liftoff_arm_timeout_sec='6.0',
    control_mode='mhe', thrust_map_fault_gain='1.0', lumped_force_enable='true',
    l1_a_gain='10.0', l1_omega_c='0.5', grip_drop_after_sec='45.0',
    drop_unresolved_timeout_sec='12.0', drop_unresolved_stop_sec='3.0',
    grip_dynamic_after_lift='true', grip_dyn_r='10.0', grip_dyn_w='0.1415',
    grip_dyn_ramp='9.36', grip_dyn_yaw_ramp='true', grip_dyn_dz='0.8',
    traj_scale_weights='false', grip_drop_at_fig8_tip='true',
    grip_uncommanded_loss='false', attach_window_sec='140.0',
    omega_scale_enable='true', omega_scale_source='djest',
    omega_scale_tau='0.5', omega_scale_cap='5.0')

SCEN = {
    'nominal': dict(),
    'd4': dict(detach_delay=4.0),
    'loss': dict(params=dict(grip_uncommanded_loss='true')),
    'jam': dict(jam=True),
    'k09': dict(params=dict(thrust_map_fault_gain='0.9')),
    'v4': dict(params=dict(grip_dyn_w='0.283', grip_dyn_ramp='0.0')),
    'p15': dict(payload=0.15),
}[scen]
params = (dict(SCEN.get('params', {})) if os.environ.get('MINIMAL') == '1'
          else dict(MAINLINE, **SCEN.get('params', {})))
PAYLOAD = SCEN.get('payload', 0.30)
DETACH_DELAY = SCEN.get('detach_delay', 0.0)
JAM = SCEN.get('jam', False)
T_END = SCEN.get('t_end', 175.0)

# ------------------------------------------------------------ 伪 ROS 外壳
T_NS = 0
REC = dict(pubs=[], logs=[], calls=[])
SUBS = {}
TIMERS = []


def now_sec():
    return T_NS * 1e-9


class FakeClock:
    def now(self):
        return Time(nanoseconds=T_NS)


_CLOCK = FakeClock()


class RecLogger:
    def _rec(self, lvl, m):
        REC['logs'].append((T_NS, lvl, str(m)))

    def info(self, m, **kw): self._rec('I', m)
    def warn(self, m, **kw): self._rec('W', m)
    warning = warn
    def error(self, m, **kw): self._rec('E', m)
    def debug(self, m, **kw): pass


_LOGGER = RecLogger()
LAST = {}


class RecPub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        if self.topic == '/parameter_events':
            return
        REC['pubs'].append((T_NS, self.topic, repr(message_to_ordereddict(msg))))
        LAST[self.topic] = msg


class FakeFuture:
    def add_done_callback(self, cb):
        pass

    def done(self):
        return False


STATE = dict(mode='AUTO.LOITER', armed=False, mode_at=None, arm_at=None)


class FakeClient:
    def __init__(self, name):
        self.name = name

    def service_is_ready(self):
        return True

    def wait_for_service(self, timeout_sec=None):
        return True

    def call_async(self, req):
        REC['calls'].append((T_NS, self.name, repr(message_to_ordereddict(req))))
        if self.name == '/mavros/set_mode' and STATE['mode_at'] is None:
            STATE['mode_at'] = T_NS + 200_000_000
        if self.name == '/mavros/cmd/arming' and STATE['arm_at'] is None:
            STATE['arm_at'] = T_NS + 200_000_000
        return FakeFuture()


class FakeTimer:
    def __init__(self, period, cb):
        self.period_ns = int(round(period * 1e9))
        self.cb = cb
        self.next_ns = T_NS + self.period_ns


def _create_publisher(self, msg_type, topic, qos, **kw):
    return RecPub(topic)


def _create_subscription(self, msg_type, topic, cb, qos, **kw):
    SUBS[topic] = cb
    return object()


def _create_timer(self, period, cb, **kw):
    t = FakeTimer(period, cb)
    TIMERS.append(t)
    return t


Node.get_clock = lambda self: _CLOCK
Node.get_logger = lambda self: _LOGGER
Node.create_publisher = _create_publisher
Node.create_subscription = _create_subscription
Node.create_timer = _create_timer
Node.create_client = lambda self, srv_type, name, **kw: FakeClient(name)


class FakeTime:
    """替换节点模块里的 time:墙钟/单调钟都走仿真时钟。"""
    @staticmethod
    def time():
        return now_sec()

    @staticmethod
    def monotonic():
        return now_sec()


# ------------------------------------------------------------ 载入被测节点
import offboard_test_acados  # noqa: E402,F401
spec = importlib.util.spec_from_file_location(
    'offboard_test_acados._nmpc_under_test', node_file)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
mod.time = FakeTime

args = ['harness', '--ros-args']
for k, v in params.items():
    args += ['-p', f'{k}:={v}']
rclpy.init(args=args)
node = mod.AcadosNMPCNode()
REC['init_log_count'] = len(REC['logs'])

from mavros_msgs.msg import State  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402
from actuator_msgs.msg import Actuators  # noqa: E402
from offboard_test_acados.payload_estimate import PayloadEstimate  # noqa: E402

MOTOR_CONSTANT = 8.54858e-06
G = 9.81
M_B = 2.0643
BOX_LOCAL = np.array([1.0 + 0.021, -0.017])   # EKF 本地系相对世界系的漂移

# ------------------------------------------------------------ 合成被控对象
x = np.zeros(13)
x[6] = 1.0
x[2] = 0.05
seg = None          # (t0_ns, x_from, x_to) NMPC 接管后的线性插值段
attached = False
airborne = False
attach_z = None
detach_at = None
detached = False
est = dict(m=M_B, dj=0.0, conf=1.0)
last_u_count = 0


def quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def plant_step(dt):
    global x, seg, last_u_count
    if not node_started():
        sp = LAST.get('/mavros/setpoint_position/local')
        if sp is None or not (STATE['armed'] and STATE['mode'] == 'OFFBOARD'):
            return
        p_sp = np.array([sp.pose.position.x, sp.pose.position.y, sp.pose.position.z])
        v = np.clip((p_sp - x[0:3]) / 1.0, -1.0, 1.0)
        x[3:6] = v
        x[0:3] = x[0:3] + v * dt
        x[6:10] = [1.0, 0.0, 0.0, 0.0]
        x[10:13] = 0.0
        return
    n_u = U_COUNT[0]
    if n_u != last_u_count:
        last_u_count = n_u
        xp = np.asarray(node.solver.get(0, 'x'), dtype=float)
        if xp.shape == (13,) and np.all(np.isfinite(xp)):
            seg = (T_NS, x.copy(), xp.copy())
    if seg is not None:
        t0, xa, xb = seg
        s = min(max((T_NS - t0) * 1e-9 / 0.05, 0.0), 1.0)
        x = xa + s * (xb - xa)
        x[6:10] = quat_norm(x[6:10])


def node_started():
    return bool(getattr(node, 'nmpc_started', False))


U_COUNT = [0]
_orig_pub = RecPub.publish


def _pub_hook(self, msg):
    _orig_pub(self, msg)
    if self.topic == '/acados_nmpc/u_opt':
        U_COUNT[0] += 1
    if self.topic == '/gripper/enable':
        _on_enable(bool(msg.data))
    if self.topic == '/gripper/release' and bool(msg.data):
        _detach()


RecPub.publish = _pub_hook


def _on_enable(v):
    global attached, attach_z, detach_at
    if v and not attached and not detached:
        attached = True
        attach_z = float(x[2])
    elif not v and attached and not JAM:
        detach_at = T_NS + int(DETACH_DELAY * 1e9)


def _detach():
    global attached, airborne, detached
    attached = False
    airborne = False
    detached = True


def m_true():
    return M_B + (PAYLOAD if airborne else 0.0)


def noise(k, amp, f):
    return amp * math.sin(2 * math.pi * f * now_sec() + k)


def send_odom():
    msg = Odometry()
    msg.header.stamp = _CLOCK.now().to_msg()
    q = quat_norm(x[6:10] + np.array([0.0, noise(1, 2e-4, 1.3), noise(2, 2e-4, 0.7), 0.0]))
    p_ = x[0:3] + np.array([noise(3, 1e-3, 0.9), noise(4, 1e-3, 1.1), noise(5, 1e-3, 0.5)])
    # odom 速度是机体系:世界速度转到机体
    w, xq, yq, zq = q
    R = np.array([
        [1 - 2*(yq*yq + zq*zq), 2*(xq*yq - zq*w), 2*(xq*zq + yq*w)],
        [2*(xq*yq + zq*w), 1 - 2*(xq*xq + zq*zq), 2*(yq*zq - xq*w)],
        [2*(xq*zq - yq*w), 2*(yq*zq + xq*w), 1 - 2*(xq*xq + yq*yq)]])
    vb = R.T @ (x[3:6] + np.array([noise(6, 5e-3, 1.7), noise(7, 5e-3, 1.9), noise(8, 5e-3, 2.3)]))
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, p_)
    msg.pose.pose.orientation.w, msg.pose.pose.orientation.x = float(q[0]), float(q[1])
    msg.pose.pose.orientation.y, msg.pose.pose.orientation.z = float(q[2]), float(q[3])
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, vb)
    om = x[10:13] + np.array([noise(9, 2e-3, 3.1), noise(10, 2e-3, 2.9), noise(11, 1e-3, 1.5)])
    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, om)
    SUBS['/mavros/local_position/odom'](msg)


def thrust_now():
    on = STATE['armed'] and (x[2] > 0.06 or node_started() or
                             LAST.get('/mavros/setpoint_position/local') is not None)
    return m_true() * G * (1.0 + noise(12, 3e-3, 2.7)) if on else 0.0


def send_motor():
    T = thrust_now()
    w = math.sqrt(max(T, 0.0) / (4 * MOTOR_CONSTANT))
    msg = Actuators()
    msg.velocity = [w * (1 + noise(13 + i, 1e-3, 4.1 + i)) for i in range(4)]
    cb = SUBS.get('/x500_0/command/motor_speed')
    if cb:
        cb(msg)


def send_imu():
    msg = Imu()
    T = thrust_now()
    msg.linear_acceleration.z = float(T / m_true()) if T > 0 else 0.0
    cb = SUBS.get('/mavros/imu/data')
    if cb:
        cb(msg)


def send_estimate():
    loaded = airborne
    a = 1.0 - math.exp(-0.1 / 0.4)
    est['m'] += a * (m_true() - est['m'])
    est['dj'] += a * ((0.0186 * PAYLOAD / 0.3 if loaded else 0.0) - est['dj'])
    est['conf'] += a * ((0.02 if loaded else 1.0) - est['conf'])
    pe = PayloadEstimate(
        m_total=est['m'] + noise(20, 2e-3, 0.37), s_xy=np.array([noise(21, 1e-3, 0.2), 0.0]),
        c_xy=np.zeros(2), dJ_diag=np.array([est['dj'], est['dj'], 0.0]),
        J_diag=np.array([0.0142 + est['dj'], 0.0142 + est['dj'], 0.026]),
        no_payload_confidence=float(np.clip(est['conf'], 0.0, 1.0)),
        healthy=True, solution_age_sec=0.05)
    cb = SUBS.get('/mhe/payload_estimate')
    if cb:
        cb(Float64MultiArray(data=[float(v) for v in pe.as_array()]))


def send_target_rel():
    cb = SUBS.get('/gripper/target_rel')
    if cb:
        rel = BOX_LOCAL - x[0:2]
        cb(Float64MultiArray(data=[float(rel[0]), float(rel[1]), float(-x[2])]))


def send_state():
    msg = State()
    msg.connected = True
    msg.armed = STATE['armed']
    msg.mode = STATE['mode']
    SUBS['/mavros/state'](msg)


# ------------------------------------------------------------ 主循环(10 ms 节拍)
TICK = 10_000_000
k = 0
while T_NS < int(T_END * 1e9):
    T_NS += TICK
    k += 1
    if STATE['mode_at'] is not None and T_NS >= STATE['mode_at']:
        STATE['mode'] = 'OFFBOARD'
    if STATE['arm_at'] is not None and T_NS >= STATE['arm_at']:
        STATE['armed'] = True
    if detach_at is not None and T_NS >= detach_at and attached:
        _detach()
    if attached and not airborne and x[2] > attach_z + 0.03:
        airborne = True
    plant_step(TICK * 1e-9)
    if k % 100 == 1:
        send_state()
    send_motor()
    send_imu()
    if k % 2 == 0:
        send_odom()
    if k % 5 == 0:
        send_target_rel()
    if k % 10 == 0:
        send_estimate()
    for t in TIMERS:
        if T_NS >= t.next_ns:
            t.next_ns += t.period_ns
            t.cb()

REC['events'] = dict(attached=attached, airborne=airborne, detached=detached)
with open(out_path, 'wb') as f:
    pickle.dump(REC, f)
nl = len(REC['logs'])
print(f'[harness] {scen}: pubs={len(REC["pubs"])} logs={nl} calls={len(REC["calls"])} '
      f'solves={U_COUNT[0]} events={REC["events"]}')
