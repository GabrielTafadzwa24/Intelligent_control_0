#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import BatteryStatus, VehicleStatus, VehicleGlobalPosition
from px4_msgs.msg import VehicleCommand, OffboardControlMode
import numpy as np
import time
from geometry_msgs.msg import PoseStamped
import math

class BatteryMonitorNode(Node):
    def __init__(self):
        super().__init__('battery_monitor_node')

        # QoS profile for reliable communication
        uxrQoS_pub = QoSProfile(
            # """QoS settings for publishers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            history = HistoryPolicy.KEEP_LAST,
            depth = 0
        )
        uxrQoS_sub = QoSProfile(
            # """QoS settings for subscribers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

        # parameters
        self.declare_parameter('battery_critical_threshold', 20.0)  # battery percentage to trigger RTH
        self.declare_parameter('battery_warning_threshold', 30.0)   # battery percentage to warn
        self.declare_parameter('home_position_lat', 0.0)    # home latitude
        self.declare_parameter('home_position_lon', 0.0)    # home longitude
        self.declare_parameter('home_position_alt', 0.0)    # home altitude
        self.declare_parameter('solar_power_estimation', True)  # enable solar power estimation
        self.declare_parameter('min_solar_power', 5.0)  # minimum solar power in watts
        self.declare_parameter('max_solar_power', 100.0) # maximum solar power in watts
        self.declare_parameter('energy_per_meter', 0.05)    # energy needed per meter in Wh

        # get parameter values
        self.battery_critical = self.get_parameter('battery_critical_threshold').value
        self.bettery_warning = self.get_parameter('battery_warning_threshold').value
        self.home_lat = self.get_parameter('home_position_lat').value
        self.home_lon = self.get_parameter('home_position_lon').value
        self.home_alt = self.get_parameter('home_position_alt').value
        self.solar_power_estimation = self.get_parameter('solar_power_estimation').value
        self.min_solar_power = self.get_parameter('min_solar_power').value
        self.max_solar_power = self.get_parameter('max_solar_power').value
        self.energy_per_meter = self.get_parameter('energy_per_meter').value

        # initialize internal variables
        self.battery_remaining = 100.0
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.vehicle_armed = False
        self.rth_initiated = False
        self.vehicle_mode = 0
        self.timestamp = 0
        self.last_battery_warn_time = 0
        self.home_position_set = False

        # subscribers
        self.battery_status_sub = self.create_subscription(
            BatteryStatus, '/fmu/out/battery_status', self.battery_status_callback, uxrQoS_sub)
        
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, uxrQoS_sub)
        
        self.global_position_sub = self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.global_position_callback, uxrQoS_sub)
        

        # publishers
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', uxrQoS_pub)
        

        # timer for regular system checks
        self.timer = self.create_timer(1.0, self.system_check)

        self.get_logger().info('Battery Monitor Node initialized')

    def battery_status_callback(self, msg):
        self.battery_remaining = msg.remaining * 100    # convert to percentage
        self.timestamp = msg.timestamp

        # estimate solar power input (simplified model)
        if self.solar_power_estimation:
            voltage = msg.voltage_v
            current = msg.current_a
            discharge_current = max(0, current) # current when discharging (positive in PX4)

            if current < 0: # negative current in PX4 means charging
                self.solar_power_current = -current * voltage   # power in watts
            else:
                self.solar_power_current = 0.0

            self.get_logger().debug(f'Solar power: {self.solar_power_current:.2f}W, Battery: {self.battery_remaining:.1f}%')

    
    def vehicle_status_callback(self, msg):
        self.vehicle_armed = msg.armin_state == 2   # 2 = armed
        self.vehicle_mode = msg.nav_state

        # if this is the first time we're armed, and we don't have home setppoint, set home position
        if self.vehicle_armed and not self.home_position_set and self.home_lat == 0.0 and self.home_lon == 0.0:
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon
            self.home_alt = self.current_alt
            self.home_position_set = True

            self.get_logger().info(f'Home position is set: Lat: {self.home_lat}, Lon: {self.home_lon}, Alt: {self.home_alt}')

    def global_position_callback(self, msg):
        self.current_lat = msg.lat
        self.current_lon = msg.lon
        self.current_alt = msg.alt

    def system_check(self):
        if not self.vehicle_armed:
            return
        
        # calculate distance to home
        distance_to_home = self.calculate_distance_to_home()

        # calculate energy needed to return home
        energy_needed_wh = distance_to_home * self.energy_per_meter

        # calculating current energy available in battery (rough estimate)
        # for a Lithium Iron Phosphate (LiFePO4) battery
        battery_capacity_wh = 108 * 13  # 108Ah * 13V = 1404Wh
        available_energy_wh = battery_capacity_wh * (self.battery_remaining / 100.0)

        # adjust for solar power if available
        solar_contribution = 0.0
        if self.solar_power_current > 0:
            # rough estimate of time to home in hours (assuming 1.5 m/s speed)
            time_to_home_hrs = distance_to_home / (1.5 * 3600)
            solar_contribution = self.solar_power_current * time_to_home_hrs # solar energy duuring return

        total_available_energy = available_energy_wh + solar_contribution

        # safety margin (20%)
        energy_with_margin = energy_needed_wh * 1.2

        self.get_logger().debug(
            f'Battery: {self.battery_remaining:.1f}%, ' +
            f'Distance home: {distance_to_home:1f}m, ' +
            f'Energy needed: {energy_with_margin:.1f}Wh, ' +
            f'Avaiilable: {total_available_energy:.1f}Wh'
        )

        # critical condition - initial RTH when:
        # 1. Battery below critical threshold, or
        # 2. Energy required to get home exceeds available energy
        initial_rth = False

        if self.battery_remaining <= self.battery_critical:
            self.get_logger().warn(f'Battery critical ({self.battery_remaining::.1f}%)! Initiating return to home.')
            initial_rth = True
        elif self.battery_remaining <= self.battery_warning:
            # only warn every 30 seconds to avoid log spam
            current_time = time.time() 
            if current_time - self.last_battery_warn_time > 30:
                self.get_logger().warn(
                    f'Battery warning ({self.battery_remaining:.1f}%)! ' +
                    f'Return energy margin: {total_available_energy - energy_with_margin:.1f}Wh'
                )
                self.last_battery_warn_time = current_time

        if initial_rth and not self.rth_initiated:
            self.return_to_home()
            self.rth_initiated = True

    def return_to_home(self):
        """commanding the vehicle to return to home position"""
        self.get_logger().info('Commanding return to home')

        vehicle_command = VehicleCommand()
        vehicle_command.timestamp = self.timestamp
        vehicle_command.param1 = 0.0    #return to home mode
        vehicle_command.param2 = 0.0
        vehicle_command.param3 = 0.0
        vehicle_command.param4 = 0.0
        vehicle_command.param5 = 0.0
        vehicle_command.param6 = 0.0
        vehicle_command.param7 = 0.0
        vehicle_command.command = 22    # NAV_RETURN_TO_LAUNCH command
        vehicle_command.target_system = 1
        vehicle_command.target_component = 1
        vehicle_command.source_system = 1
        vehicle_command.source_component = 1
        vehicle_command.from_external = True

        self.vehicle_command_pub.publish(vehicle_command)

        # also publish a log message for operators
        self.get_logger().warn(
            'RETURNING TO HOME due to energy considerations. ' + 
            f'battery: {self.battery_remaining:.1f}%, Distance: {self.calculate_distance_to_home():.1f}m'
        )

    def calculate_distance_to_home(self):
        """calculatig the great circle distance between current position and home"""

        # convert to radians
        lat1 = math.radians(self.current_lat)
        lon1 = math.radians(self.current_lon)
        lat2 = math.radians(self.home_lat)
        lon2 = math.radians(self.home_lon)

        # Havesin formula
        dlon = lon2 - lon2
        dlat = lat2 - lat2
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000 # radius of earth in meters

        # calculate horizontal distance
        horizontal_distance = c * r

        # calculate vertical distance (altitude difference)
        vertical_distance = abs(self.home_alt - self.current_alt)

        # calculate 3D distance using pythogoras
        total_distance = math.sqrt(horizontal_distance**2 + vertical_distance**2)

        return total_distance

def main(args=None):
    rclpy.init(args=args)
    battery_monitor_node = BatteryMonitorNode()
    try:
        rclpy.spin(battery_monitor_node)
    except KeyboardInterrupt:
        battery_monitor_node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        battery_monitor_node.get_logger().error(f'Error: {e}')
    finally:
        battery_monitor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()