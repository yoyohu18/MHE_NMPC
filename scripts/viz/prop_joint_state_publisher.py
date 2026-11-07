#!/usr/bin/env python3
# 把 PX4 实际发给 Gazebo 电机模型的转速指令积分成桨叶转角,驱动 URDF 里
# rotor_0..3_joint(continuous)的 /joint_states,让 RViz 里的桨叶真的转起来。
#
# 数据来源: gz-transport 话题 /x500_payload_0/command/motor_speed(gz.msgs.Actuators,
# velocity 字段=4个电机的角速度 rad/s),用 ros_gz_bridge 桥接成同名 ROS2 话题
# (actuator_msgs/msg/Actuators)——见 run_sitl_acados.sh 里起的那个 bridge
# 进程。注意:这是 PX4 发出的"指令"角速度,不是 Gazebo 内部真正积分出来的桨叶
# 角度本身(那个量没有被另外发布出来),这里自己在 ROS2 侧重新积分一份,纯
# 视觉用,角度数值跟 Gazebo 渲染出来的桨叶不会逐帧像素对齐,但转向、转速量级
# 是对的。
#
# 旋转方向:rotor_0/1 用 ccw 桨叶 mesh,绕 +Z 轴(右手定则)转才是视觉上的
# "逆时针(从上往下看)";rotor_2/3 用 cw 桨叶 mesh,要绕 -Z 转。PX4 发出来的
# velocity 数值本身只是个非负的转速大小,不含方向信息(方向是电机接线决定
# 的,不在这条消息里),方向号在这里按 model.sdf 里每个 rotor 用的桨叶 mesh
# 手动指定。

import rclpy
from rclpy.node import Node
from actuator_msgs.msg import Actuators
from sensor_msgs.msg import JointState

JOINT_NAMES = ['rotor_0_joint', 'rotor_1_joint', 'rotor_2_joint', 'rotor_3_joint']
# rotor_0/1 = ccw(+Z 转向为正),rotor_2/3 = cw(-Z 转向,取负号)
SPIN_SIGN = [1.0, 1.0, -1.0, -1.0]
PUBLISH_RATE_HZ = 50.0


class PropJointStatePublisher(Node):
    def __init__(self):
        super().__init__('prop_joint_state_publisher')

        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.angle = [0.0, 0.0, 0.0, 0.0]

        # 话题名做成参数,跟 mhe_node 的 motor_speed_topic 同一套约定:
        # mass_changer 场景是 x500_payload_0,夹爪场景是空机 x500_0。默认保持
        # 原值,不影响现有 run_sitl_acados.sh。之前硬编码 x500_payload_0 时,
        # 夹爪场景的 bridge 发布的是 x500_0,这里订阅不到,RViz 里桨叶就不转。
        self.declare_parameter('motor_speed_topic',
                               '/x500_0/command/motor_speed')
        motor_topic = self.get_parameter('motor_speed_topic').value

        self.motor_speed_sub = self.create_subscription(
            Actuators, motor_topic, self.motor_speed_cb, 10)

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.dt = 1.0 / PUBLISH_RATE_HZ
        self.timer = self.create_timer(self.dt, self.timer_cb)

    def motor_speed_cb(self, msg):
        if len(msg.velocity) == 4:
            self.velocity = list(msg.velocity)

    def timer_cb(self):
        for i in range(4):
            self.angle[i] += SPIN_SIGN[i] * self.velocity[i] * self.dt

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES
        js.position = list(self.angle)
        self.joint_state_pub.publish(js)


def main():
    rclpy.init()
    node = PropJointStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
