#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
import time


class MissionControl(Node):
    """Node for controlling the vehicle through MAVROS interface."""
    
    def __init__(self) -> None:
        super().__init__('mission_control')
        
        # QoS profile for subscribers
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Create subscription for vehicle state
        self.state_sub = self.create_subscription(
            State, 
            '/mavros/state', 
            self.state_callback, 
            qos_sub
        )
        
        # Service clients for vehicle commands
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        # Vehicle state tracking
        self.current_state = State()
        self.was_armed = False
        self.returned_home = False
        self.last_command_time = 0.0
        self.command_timeout = 5.0  # Timeout in seconds
        self.command_retries = 0
        self.max_retries = 3
        
        # Timer for periodic status check and commands
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Wait for service connections
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting...')
        
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set mode service not available, waiting...')
        
        self.get_logger().info("MAVROS Mission Control node started.")

    def state_callback(self, msg):
        """Callback to update vehicle state information."""
        # Check if vehicle has just disarmed (was armed, now disarmed)
        if self.was_armed and not msg.armed:
            # Check if it's in RTL or LAND mode, which would indicate it returned home
            if msg.mode in ['AUTO.RTL', 'AUTO.LAND']:
                self.returned_home = True
                self.get_logger().info("Vehicle has returned home and disarmed.")
        
        # Update armed status tracking
        self.was_armed = msg.armed
        
        # Log mode changes
        if hasattr(self, 'previous_mode') and self.previous_mode != msg.mode:
            self.get_logger().info(f"Vehicle mode changed from {self.previous_mode} to {msg.mode}")
        
        self.previous_mode = msg.mode
        self.current_state = msg

    def set_mission_mode(self):
        """Set the vehicle to mission mode using MAVROS services."""
        self.command_retries += 1
        self.get_logger().info(f"Switching to Mission Mode... (attempt {self.command_retries}/{self.max_retries})")
        
        # Create SetMode request
        request = SetMode.Request()
        request.base_mode = 0  # Ignore base_mode
        request.custom_mode = 'AUTO.MISSION'
        
        # Send request
        future = self.set_mode_client.call_async(request)
        future.add_done_callback(self.set_mode_callback)
        
        self.last_command_time = time.time()

    def set_mode_callback(self, future):
        """Callback for set_mode service response."""
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info("Mode change request accepted.")
                self.command_retries = 0
            else:
                self.get_logger().warn("Mode change request rejected.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def arm_vehicle(self, arm):
        """Arm or disarm the vehicle."""
        request = CommandBool.Request()
        request.value = arm
        
        future = self.arming_client.call_async(request)
        if arm:
            future.add_done_callback(self.arm_callback)
        else:
            future.add_done_callback(self.disarm_callback)

    def arm_callback(self, future):
        """Callback for arming service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Vehicle armed successfully.")
            else:
                self.get_logger().warn("Failed to arm vehicle.")
        except Exception as e:
            self.get_logger().error(f"Arming service call failed: {e}")

    def disarm_callback(self, future):
        """Callback for disarming service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Vehicle disarmed successfully.")
            else:
                self.get_logger().warn("Failed to disarm vehicle.")
        except Exception as e:
            self.get_logger().error(f"Disarming service call failed: {e}")

    def timer_callback(self):
        """Check vehicle state and send mission mode command if needed."""
        # Check if we need to retry commands
        if (time.time() - self.last_command_time > self.command_timeout and 
            self.command_retries > 0 and self.command_retries < self.max_retries):
            # Retry setting mission mode
            if self.current_state.mode != 'AUTO.MISSION':
                self.set_mission_mode()
        
        # If we haven't sent any commands yet or max retries reached, try once
        elif self.current_state.mode != 'AUTO.MISSION' and self.command_retries == 0:
            self.set_mission_mode()
        elif self.current_state.mode == 'AUTO.MISSION':
            if self.command_retries > 0:
                self.get_logger().info("Vehicle is now in Mission Mode.")
                self.command_retries = 0
            # No need for else clause as we don't want to spam the log


def main():
    """Initialize and run the MAVROS mission control node."""
    try:
        rclpy.init()
        control_node = MissionControl()
        
        print("MAVROS Mission Control node spinning. Press Ctrl+C to exit.")
        rclpy.spin(control_node)
    except Exception as e:
        print(f"Error in MAVROS Mission Control node: {e}")
    finally:
        # Ensure proper cleanup
        if 'control_node' in locals():
            control_node.destroy_node()
        rclpy.shutdown()
        print("MAVROS Mission Control node has been shut down.")


if __name__ == '__main__':
    main()