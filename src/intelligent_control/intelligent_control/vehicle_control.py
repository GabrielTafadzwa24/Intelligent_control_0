#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand, VehicleStatus


class VehicleControl(Node):
    """Node for controlling the vehicle"""
    def __init__(self) -> None:
        super().__init__('vehicle_control')

        # QoS profile for reliable communication
        uxrQoS_pub = QoSProfile(
            # """QoS settings for publishers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            history = HistoryPolicy.KEEP_LAST,
            depth = 0,
        )

        uxrQoS_sub = QoSProfile(
            # """QoS settings for subscribers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', uxrQoS_pub)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, uxrQoS_sub)
        self.get_logger().info("vehicle control node started.")

        self.timer = self.create_timer(1.0, self.timer_callback)

        # Vehicle status tracking
        self.vehicle_status = VehicleStatus()

    def vehicle_status_callback(self, msg):
        """Callback to update vehicle status."""
        self.vehicle_status = msg

    def publish_vehicle_command(self, command, **params):
        """Vehicle command..."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.vehicle_command_publisher.publish(msg)

    def set_mission_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0)
        self.get_logger().info("Switching to Mission Mode...")

    def timer_callback(self):
        """Check vehicle status and send mission mode command if needed."""
        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
            self.set_mission_mode()
        else:
            self.get_logger().info("Vehicle is already in Mission Mode.")


def main():
    rclpy.init()
    control_node = VehicleControl()
    rclpy.spin(control_node)
    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Nodes terminated.")