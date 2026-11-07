#!/usr/bin/env python3
# 把实时里程计广播成 map->base_link 的 TF,配合 urdf/gripper/x500.urdf + robot_state_publisher
# 在 RViz 的 RobotModel 显示项里渲染出完整机身(机架+4个电机/桨叶)。取代之前
# 单一 mesh 的 Marker 方案(drone_marker_publisher.py)——同一份 odom,换成更
# "标准"的 URDF+TF 渲染路径,能看到电机/桨叶而不仅是中央机身,且姿态補偿直接
# 写在 URDF 的 visual origin 里,不需要在这里手算四元数 hack。

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class DroneTFBroadcaster(Node):
    def __init__(self):
        super().__init__('drone_tf_broadcaster')
        self.br = TransformBroadcaster(self)

        mavros_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/local_position/odom',
            self.odom_cb, mavros_sensor_qos)

    def odom_cb(self, msg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

        # 2026-08-21:给 RViz 视角跟随用的"水平跟随帧"。把 Orbit 的 Target Frame
        # 直接设成 base_link 也能跟住,但视图会跟着机体 roll/pitch 一起翻——4 m/s
        # 机动时倾角十几度,地平线一直在歪。chase_link 只继承位置、姿态恒为单位
        # 四元数,所以跟随平滑、地平线始终水平。纯可视化,不进任何控制/估计链路。
        c = TransformStamped()
        c.header.stamp = t.header.stamp
        c.header.frame_id = 'map'
        c.child_frame_id = 'chase_link'
        c.transform.translation = t.transform.translation
        c.transform.rotation.w = 1.0
        self.br.sendTransform(c)


def main():
    rclpy.init()
    node = DroneTFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
