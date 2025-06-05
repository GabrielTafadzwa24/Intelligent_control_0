#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool
import time


class ArmVehicle(Node):
    """Node to arm the vehicle using MAVROS"""
    
    def __init__(self):
        super().__init__("arm_vehicle")
        
        # Create service client for arming
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        
        # Wait for service to become available
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting...')
        
        # Arm immediately when node starts
        self.arm_vehicle()
        
        # Optional: Create a timer for retrying arming if needed
        # self.retry_count = 0
        # self.max_retries = 5
        # self.timer = self.create_timer(2.0, self.retry_arm)
        
        self.get_logger().info("Arm Vehicle node initialized, sending arm command...")
    
    def arm_vehicle(self):
        """Arm the vehicle using MAVROS service."""
        request = CommandBool.Request()
        request.value = True  # True for arm, False for disarm
        
        # Send the service request asynchronously
        future = self.arm_client.call_async(request)
        # Add callback to handle the response
        future.add_done_callback(self.arm_callback)
        
        self.get_logger().info("Arming request sent...")
    
    def arm_callback(self, future):
        """Callback for arming service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Vehicle armed successfully!")
            else:
                self.get_logger().warn("Failed to arm vehicle. Vehicle may be in a state that prevents arming.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
    
    # Optional: Method to retry arming if the first attempt fails
    # def retry_arm(self):
    #     """Retry arming if needed."""
    #     if self.retry_count < self.max_retries:
    #         self.get_logger().info(f"Retrying arm command (attempt {self.retry_count + 1}/{self.max_retries})...")
    #         self.arm_vehicle()
    #         self.retry_count += 1
    #     else:
    #         self.get_logger().warn("Maximum retry attempts reached. Stopping retries.")
    #         self.timer.cancel()


def main():
    rclpy.init()
    
    try:
        arm_node = ArmVehicle()
        rclpy.spin(arm_node)
    except KeyboardInterrupt:
        print("Node terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        if 'arm_node' in locals():
            arm_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()