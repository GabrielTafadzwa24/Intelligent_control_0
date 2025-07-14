#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from intelligent_control_msgs.msg import MyFuzzyOutput
import pandas as pd 
from datetime import datetime
import os
import signal
import sys

class DataLogger(Node):
    """Node that logs left motor speed and right motor speed data to an Excel file."""

    def __init__(self):
        super().__init__('data_logger')

        # Initialize communication infrastracture
        self._init_quality_of_service_profiles()
        self._init_subscribers()

        # Initialize data storage
        self._init_data_storage()

        # Control flags
        self._init_control_flags()

        # Configuration parameters
        self._init_navigation_parameters()

        # Validation parameters
        self._init_validation_parameters()
        
        # Set up signnal handlers for graceful shutdown
        self._init_signal_handler()
        
        self.get_logger().info(f"Data logger initialized. Will start logging at waypoint {self.start_logging_waypoint}")

    def _init_quality_of_service_profiles(self):
        """QoS settings for subscribers."""
        self.uxrqos_sub = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

    def _init_subscribers(self):
        """" Initializing subscriber for combined data."""
        self.nav_data_sub = self.create_subscription(
            MyFuzzyOutput, '/fuzzy_navigation/output', self.nav_data_callback, self.uxrqos_sub)
        
    def _init_data_storage(self):
        """Initializing data storage."""
        self.data = {
            'timestamp' : [],
            'wp_index' : [],
            'distance_to_waypoint' : [],
            'cross_track_error' : [],
            'wave_condition' : [],
            'left_motor_speed' : [],
            'right_motor_speed' : []
        }

    def _init_control_flags(self):
        """Initializing control flags."""
        self.logging_active = False
        self.mission_complete = False
        self.data_saved = False
        self.max_waypoint_seen = -1

    def _init_navigation_parameters(self):
        """Initializing navigation parameters and declare ROS parameters."""
        self.declare_parameter('start_waypoint', 1)
        self.declare_parameter('log_directory', 'navigation_logs')
        self.declare_parameter('reset_threshold', 5)

        self.start_logging_waypoint = self.get_parameter('start_waypoint').value
        self.log_directory = self.get_parameter('log_directory').value
        self.reset_threshold = self.get_parameter('reset_threshold').value

    def _init_validation_parameters(self):
        """Initializing validation parameters."""
        if self.start_logging_waypoint < 0:
            self.get_logger().warn(f"Invalid start waypoint {self.start_logging_waypoint}, setting to 0")
            self.start_logging_waypoint = 0

        if self.reset_threshold < 1:
            self.get_logger().warn(f"Invalid reset threshold {self.reset_threshold}, setting to 5")
            self.reset_threshold = 5

    def _init_signal_handler(self):
        """Initializing signal handler."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def nav_data_callback(self, msg):
        """Handle incoming navigation status messages."""
        current_index = msg.current_wp_index
        waypoint_length = msg.waypoints_length
        self.name = msg.name

        # validate message data
        if waypoint_length <= 0:
            self.get_logger().warn("Received message with invalid waypoint length <= 0")
            return
        
        if waypoint_length < 0:
            self.get_logger().warn(f"Received message with invalid current_wp_index: {current_index}")
            return
        
        end_waypoint = waypoint_length - 1

        # Track the maximum waypoint index we've seen
        if current_index > self.max_waypoint_seen:
            self.max_waypoint_seen = current_index

        # Start logging when reaching the configured waypoint
        if (current_index >= self.start_logging_waypoint and not self.logging_active and
             not self.mission_complete and current_index < end_waypoint):
            
            self.logging_active = True
            self.get_logger().info(f"Started logging at waypoint index {current_index}")

        # Check if logging should stop
        if self.logging_active and not self.mission_complete:
            # Stop if we have a specific end waypoint and we've reached it
            if current_index >= end_waypoint:
                self.mission_complete = True
                self.get_logger().info(f"Reached end waypoint {end_waypoint}, stopping logging")
                self._save_data_safely()
                return
            
            # Stop if waypoint index decreased significantly (mission restart/reset)
            if (self.max_waypoint_seen > self.reset_threshold and current_index < (self.max_waypoint_seen - self.reset_threshold)):
               
               self.mission_complete = True
               self.get_logger().info(f"Waypoint index dropped from {self.max_waypoint_seen} to {current_index},  mission reset detected")
               self._save_data_safely()
               return
            
        # stop the data if logging is active
        if self.logging_active and not self.mission_complete:
            self.data['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) # millisecond precision
            self.data['wp_index'].append(current_index)
            self.data['distance_to_waypoint'].append(msg.distance_to_waypoint)
            self.data['cross_track_error'].append(msg.cross_track_error)
            self.data['wave_condition'].append(msg.wave_condition)
            self.data['left_motor_speed'].append(msg.left_motor_speed)
            self.data['right_motor_speed'].append(msg.right_motor_speed)
            

            # Log progress every 50 data points
            if len(self.data['left_motor_speed']) % 50 == 0:
                self.get_logger().info(f"Logged {len(self.data['left_motor_speed'])} data points, current waypoint: {current_index}")
            
    def _save_data_safely(self):
        """Thread-safe warapper for saving data that prevents dupliicate saves."""
        if not self.data_saved:
            self.data_saved = True
            self.save_to_excel()

    def save_to_excel(self):
        """Save collected data to Excel."""
        if not self.data['left_motor_speed']:
            self.get_logger().warn("No data to save.")
            return
  
        try:
            df = pd.DataFrame(self.data)

            # Creating filename with waypoint range and timestamp
            start_wp = self.data['wp_index'][0] if self.data['wp_index'] else 0
            end_wp = self.data['wp_index'][-1] if self.data['wp_index'] else 0
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_mission_wp{start_wp}_to_wp{end_wp}_{timestamp}.xlsx"

            # create directory if it doesn't exist
            os.makedirs(self.log_directory, exist_ok=True)
            filepath = os.path.join(self.log_directory, filename)

            # save to excel with additional formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Navigation Data', index=False)

                # add summary statistics
                summary_data = {
                    'Metric': ['Total Data Points', 'Duration (minutes)', 'Average Left Speed', 'Max Left Speed',
                               'Min Left Speed', 'Average Right Speed', 'Max Right Speed', 'Min Right Speed',
                               'Start Waypoint', 'End Waypoint'],
                    'Value': [
                        len(df),
                        self.calculate_duration_minutes(df),
                        df['left_motor_speed'].mean() if len(df) > 0 else 0,
                        df['left_motor_speed'].max() if len(df) > 0 else 0,
                        df['left_motor_speed'].min() if len(df) > 0 else 0,
                        df['right_motor_speed'].mean() if len(df) > 0 else 0,
                        df['right_motor_speed'].max() if len(df) > 0 else 0,
                        df['right_motor_speed'].min() if len(df) > 0 else 0,
                        start_wp,
                        end_wp
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            self.get_logger().info(f"Successfully saved {len(df)} data points to {filepath}")
            self.get_logger().info(f'Data collection summary: WP {start_wp} to WP {end_wp}, Duration: {self.calculate_duration_minutes(df):.2f} minutes')

        except Exception as e:
            self.get_logger().error(f"Failed to save data to Excel: {e}")

            # Try to save as CSV as fall back
            try:
                csv_filename = filename.replace('.xlsx', '.csv')
                csv_filepath = os.path.join(self.log_directory, csv_filename)
                df.to_csv(csv_filepath, index=False)
                self.get_logger().info(f"Saved data as CSV fallback: {csv_filepath}")
            except Exception as csv_e:
                self.get_logger().error(f"Failed to save CSV fallback: {csv_e}")

    def calculate_duration_minutes(self, df):
        """Calculate the duration of data collection in minutes."""
        if len(df) < 1:
            return 0.0
        
        try:
            start_time = pd.to_datetime(df['timestamp'].iloc[0])
            end_time = pd.to_datetime(df['timestamp'].iloc[-1])
            duration = (end_time - start_time).total_seconds() / 60.0
            return duration
        except:
            return 0.0
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signal gracefully."""
        self.get_logger().info(f"Received signal {signum}, shutting down gracefully...")
        if self.logging_active and not self.mission_complete:
            self._save_data_safely()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def destroy_node(self):
        """Clean up before node destruction."""
        if self.logging_active and not self.mission_complete:
            self.get_logger().info("Node being destroyed, saving data....")
            self._save_data_safely()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt recived")
    except Exception as e:
        node.get_logger().info(f"Unexpected error: {e}")
    finally:
        # Ensure data is saved before shutdown
        if (hasattr(node, 'logging_active') and node.logging_active and 
            not node.mission_complete and not node.data_saved):
            node._save_data_safely()

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()