#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from collections import deque
from scipy.spatial.transform import Rotation
import math
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, WaypointList, Waypoint, ActuatorControl, Altitude
from mavros_msgs.srv import CommandBool, SetMode, WaypointPush
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from transforms3d.euler import quat2euler
import threading
from threading import Lock
from intelligent_control_msgs.msg import MyFuzzyOutput

class FuzzyNavigation(Node):
    """Node that uses a Fuzzy Logic Controller to control thrust navigation using MAVROS."""

    def __init__(self):
        super().__init__('fuzzy_navigation_mavros')

        # Initialize communication infrastracture
        self._init_quality_of_service_profiles()
        self._init_publishers()
        self._init_subscribers()
        self._init_services()

        self.waypoint_lock = threading.Lock()   # prevent concurrent access to waypoint-related variables

        # Initializing navigation state variables
        self._init_vehicle_state()
        self._init_navigation_parameters()

        # Creating the fuzzy logic controller
        self.create_fuzzy_controller()

        # Timer to adjust mission parameters
        self.timer = self.create_timer(0.5, self.navigation_timer_callback)

        self.get_logger().info("Fuzzy Navigation MAVROS node started.")

    def _init_quality_of_service_profiles(self):
        """Initializing QoS profile for reliable communication."""
        self.uxrQoS_pub = QoSProfile(
            # """QoS settings for publishers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            history = HistoryPolicy.KEEP_LAST,
            depth = 0
        )
        self.uxrQoS_sub = QoSProfile(
            # """QoS settings for subscribers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

    def _init_publishers(self):
        """Initializing all publishers."""
        self.motor_command_publisher = self.create_publisher(
            ActuatorControl, '/mavros/actuator_control', self.uxrQoS_pub)
        
        self.nav_status_publisher = self.create_publisher(
            MyFuzzyOutput, '/fuzzy_navigation/output', self.uxrQoS_pub
        )
        
    def _init_subscribers(self):
        """Initialize subscribers."""
        self.local_position_subscriber = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.local_position_callback, self.uxrQoS_sub)


        self.global_position_subscriber = self.create_subscription(
            NavSatFix, "/mavros/global_position/global", self.global_position_callback, self.uxrQoS_sub)
        
        
        self.waypoint_subscriber = self.create_subscription(
            WaypointList, '/mavros/mission/waypoints', self.waypoint_callback, self.uxrQoS_sub)
        
        self.state_subscriber = self.create_subscription(
            State, '/mavros/state', self.state_callback, self.uxrQoS_sub)
        
        self.imu_subscriber = self.create_subscription(
            Imu, 'mavros/imu/data', self.imu_callback, self.uxrQoS_sub)

        self.altitude_subscriber = self.create_subscription(
            Altitude, '/mavros/altitude', self.altitude_callback, self.uxrQoS_sub)
        
    def _init_services(self):
        """Initialize service clients"""
        self.arm_service = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_service = self.create_client(SetMode, '/mavros/set_mode')
        self.waypoint_push_service = self.create_client(WaypointPush, '/mavros/mission/push')

    def _init_vehicle_state(self):
        """Initializing vehicle state variables"""
        self.current_position = PoseStamped()
        self.global_position = NavSatFix()
        self.vehicle_heading = 0.0     # Current heading in radians
        self.vehicle_state = State()

    def _init_navigation_parameters(self):
        """Initializing navigation parameters and declare ROS parameters."""

        # Path tracking variables
        self.distance_to_target_wp = 0.0     # Distance to waypoint
        self.distance_from_prev_wp = 0.0     # Distance from previous waypoint
        self.distance_prev_target_wp = 0.0   # Distance between waypoints
        self.cross_track_error = 0.0        # Cross track error of the vehicle
        self.wave_condition = 0.0
        self.theta_actual_path = 0.0        # The angle between waypoint path and actual path of the vehicle

        # Waypoint management  
        self.waypoints = []
        self.current_wp_index = 0
        self.previous_waypoint = (0.0, 0.0)

        # Parameters
        self.declare_parameter('~wave_compensation_gain', 30)
        self.wave_compensation_gain = self.get_parameter('~wave_compensation_gain').value
        self.declare_parameter('~altitude_smoothing_window', 10)
        self.altitude_smoothing_window = self.get_parameter('~altitude_smoothing_window').value
        self.declare_parameter('~pitch_threshold', 0.1)
        self.pitch_threshold = self.get_parameter('~pitch_threshold').value 

        # State variables
        self.altitude_history = deque(maxlen=self.altitude_smoothing_window)
        self.altitude_rate = 0.0
        self.wave_phase = 0.0         # -1: going down, 0: stable, 1: going up
        self.current_altitude = 0.0
        self.current_pitch = 0.0

        # Declare ROS parameters
        self._declare_ros_parameters()

    def _declare_ros_parameters(self):
        """Declaring all ROS parameters."""
        # Home position parameters
        self.declare_parameter('home_latitude', 0.0)
        self.declare_parameter('home_longitude', 0.0)
        self.declare_parameter('home_altitude', 0.0)

        # Debug parameters
        self.declare_parameter('debug_coordinates', False)
        self.declare_parameter('debug_navigation', False)

        # Navigation parameters
        self.declare_parameter('waypoint_reached_threshold', 5.0)
        self.waypoint_reached_threshold = self.get_parameter('waypoint_reached_threshold').value

    # Callback methods
    def local_position_callback(self, msg):
        """Callback to update vehicle position from local position topic."""
        self.current_position = msg
        self._update_vehicle_heading_from_quaternion(msg)        
        self.check_waypoint_reached()

    def _update_vehicle_heading_from_quaternion(self, msg):
        """Extracting heading from pose message quaternion."""
        q = [
            msg.pose.orientation.w, 
            msg.pose.orientation.x, 
            msg.pose.orientation.y, 
            msg.pose.orientation.z
        ]
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        _, _, self.vehicle_heading = quat2euler([q[0], q[1], q[2], q[3]])

    def global_position_callback(self, msg):
        """Callback to update vehicle global position."""
        self.global_position = msg
        self.check_waypoint_reached()

    def state_callback(self, msg):
        """Callback to update vehicle state."""
        self.vehicle_state = msg

    def imu_callback(self, msg):
        """Process IMU data to extract pitch angle."""
        # Convert quatermiom to euler angles
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        rotation = Rotation.from_quat(q)
        euler = rotation.as_euler('xyz', degrees=False)
        self.current_pitch = euler[1]  # Pitch angle

    def altitude_callback(self, msg):
        """Process barometric altitude data."""
        self.current_altitude = msg.amsl
        self.altitude_history.append(self.current_altitude)

        # Calculate altitude rate of change for wave detection
        if len(self.altitude_history) >= 2:
            dt = 0.1    # Assuming 10Hz update rate
            self.altitude_rate = (self.altitude_history[-1] - self.altitude_history[-2]) / dt

            # Determining wave phase based on altitude rate and pitch
            self.update_wave_phase()

    def update_wave_phase(self):
        """Determining wave phase based on pitch angle and altitude rate."""

        # use combination of pitch angle and altitude rate
        if abs(self.current_pitch) > self.pitch_threshold:
            if self.current_pitch > 0 and self.altitude_rate > 0.02:    # Bow up, climbing
                self.wave_phase = 1.0 # Going up wave
            elif self.current_pitch < 0 and self.altitude_rate < -0.02: # Bow down, descending
                self.wave_phase = -1.0    # Going down wave
            else:
                self.wave_phase = 0.0 # Transitional/stable
        else:
            self.wave_phase = 0.0 # Stable condition

        
    # Distance calculation method        
    def distance_calculation(self, lat1, lon1, lat2=None, lon2=None):
        """Calculate distance between two points or from current position to a point,: meters."""
        if lat2 is None or lon2 is None:
            lat2, lon2 = self.global_position.latitude, self.global_position.longitude

        R = 6371000  # metres
        rlat1 = math.radians(lat1)
        rlat2 = math.radians(lat2)

        rlat_d = math.radians(lat2 - lat1)
        rlon_d = math.radians(lon2 - lon1)

        a = (math.sin(rlat_d / 2) * math.sin(rlat_d / 2) + math.cos(rlat1) *
             math.cos(rlat2) * math.sin(rlon_d / 2) * math.sin(rlon_d / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        d = R * c
        
        return d
    
    # Waypoint handling methods        
    def waypoint_callback(self, msg):
        """
        Callback for receiving mission waypoints from MAVROS.
        
        Processes waypoint list messages, converts global coordinates to local coordinates,
        and updates the navigation path parameters.
        
        Args:
            msg (WaypointList): Message containing the list of waypoints from MAVROS
        """
        # Skip processing if message is empty
        if len(msg.waypoints) == 0:
            self.get_logger().warn("Received empty waypoint list")
            return
            
        # Log receipt of waypoints
        self.get_logger().info(f"Received waypoint list with {len(msg.waypoints)} waypoints")

        # Process waypoints
        self._process_waypoint_list(msg.waypoints)

        # Update navigation parameters based on current waypoint
        self.update_navigation_targets()

        # log current navigation status
        self._log_navigation_status()
        
    def _process_waypoint_list(self, raw_waypoints):
        """Process and filter a list of raw waypoints.
        
        Args:
            raw_waypoints (list): List of Waypoint messages from MAVROS"""
        # Store the original waypoints for potential future reference
        with self.waypoint_lock:
            self.original_waypoints = raw_waypoints.copy()
        
            # Clear existing waypoints and reset navigation state
            self.waypoints = []
            
            # Flag to track if the current waypoint was identified in the message
            current_wp_found = False
            current_wp_index = 0
            
            # Process each waypoint in the message
            for i, wp in enumerate(raw_waypoints):
                # Skip certain waypoints based on command type
                if wp.command not in [16, 22, 82, 84]:  # Navigation commands: waypoint, takeoff, spline, loiter
                    self.get_logger().debug(f"Skipping non-navigation waypoint at index {i} (cmd={wp.command})")
                    continue
                    
                # Check if this is marked as the current waypoint
                if wp.is_current:
                    current_wp_index = len(self.waypoints)
                    current_wp_found = True
                    self.get_logger().info(f"Current waypoint identified at mission index {i}")
                
                # Store local coordinates and metadata in a dictionary for richer waypoint representation
                waypoint_data = {
                    'global_coordinates': (wp.x_lat, wp.y_long, wp.z_alt),
                    'command': wp.command,
                    'param1': wp.param1,  # Hold time for waypoints
                    'param2': wp.param2,  # Acceptance radius
                    'param3': wp.param3,  # Pass through waypoint (0 = no)
                    'param4': wp.param4,  # Desired yaw angle
                    'autocontinue': wp.autocontinue,
                    'frame': wp.frame,
                    'original_index': i
                }
                
                self.waypoints.append(waypoint_data)
                self.get_logger().debug(
                    f"Added waypoint: global({wp.x_lat:.7f}, {wp.y_long:.7f}, {wp.z_alt:.2f})")
            
            # Handle empty waypoint list after filtering
            if not self.waypoints:
                self.get_logger().warn("No valid waypoints found after filtering")
                return
                
            # If no current waypoint was found, set it to the first waypoint
            if not current_wp_found:
                current_wp_index = 0
                self.get_logger().info("No current waypoint flag found, defaulting to first waypoint")
            
            # Update the current waypoint index
            self.current_wp_index = current_wp_index

    def _log_navigation_status(self):
        """Log current navigation status and targets."""
        if self.current_wp_index >= len(self.waypoints):
            return
        
        self.get_logger().debug(
            f"Navigation status: targeting waypoint {self.current_wp_index} "
            f"Vehicle heading: {math.degrees(self.vehicle_heading):.2f}°")

    def check_waypoint_reached(self):
        """Check if vehicle has reached the current waypoint and advance to the next one."""
        if self.current_wp_index >= len(self.waypoints) or not self.waypoints:
            return
    
        # Get waypoint coordinates from the dictionary structure
        target_lat, target_lon, _ = self.waypoints[self.current_wp_index]['global_coordinates']
        self.distance_to_target_wp = self.distance_calculation(target_lat, target_lon)
        
        if self.distance_to_target_wp < self.waypoint_reached_threshold:
            self._advance_to_next_waypoint()

    def _advance_to_next_waypoint(self):
        """Advance to the next waypoint in the mission."""

        # Increment with bounds checking 
        if self.current_wp_index < len(self.waypoints) - 1:
            self.current_wp_index += 1
            self.get_logger().info(f"Advanced to waypoint {self.current_wp_index}")

            # Update navigation targets for new waypoint
            self.update_navigation_targets()
        else:
            self.current_wp_index = 0   # to loop
            self.get_logger().debug("Reached final waypoint.")

    def _set_previous_waypoint(self):
        """Set the previous waypoint for path calculation."""
        if self.current_wp_index > 0:
            # Use the previous waypoint in the list
            prev_wp = self.waypoints[self.current_wp_index - 1]
            self.previous_waypoint = prev_wp['global_coordinates'][:2]  # Just x and y
        else:
            # For the first waypoint, use current position as previous
            self.previous_waypoint = (
                self.global_position.latitude,
                self.global_position.longitude
            )

    def update_navigation_targets(self):
        """ Update navigation targets based on the current waypoint index."""
        # Validate waypoint index
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().error(f"Invalid waypoint index: {self.current_wp_index}")
            return False
            
        # Get current target waypoint coordinates
        current_wp = self.waypoints[self.current_wp_index]
        
        # Set the previous waypoint for path calculations
        self._set_previous_waypoint()
        
        # Update acceptance radius based on waypoint parameter
        self._update_acceptance_radius(current_wp)
        
        return True

    def _update_acceptance_radius(self, waypoint):
        """Updating waypoint acceptance radious based on waypooint parameters."""
        self.waypoint_reached_threshold = waypoint['param2']
        if self.waypoint_reached_threshold <= 0.0:
            # If not specified, use the default parameter value
            self.waypoint_reached_threshold = self.get_parameter('waypoint_reached_threshold').value
    
    # Fuzzy controller methods
    def create_fuzzy_controller(self):
        """Initialize the fuzzy logic controller for a PX4 differential vehicle."""

        # Creating fuzzy variables
        self.fuzzy_variables = self._create_fuzzy_variables()

        # Defining membership functions
        self._define_fuzzy_membership_functions()

        # Set defuzzification methods
        self._set_defuzzification_methods()

        # Create fuzzy rules
        rules = self._create_fuzzy_rules()

        # Create control system
        system = ctrl.ControlSystem(rules)
        self.fuzzy_simulator = ctrl.ControlSystemSimulation(system)

    def _create_fuzzy_variables(self):
        """Creating fuzzy variables for inputs and outputs."""
        # Input variables
        distance_to_wp = ctrl.Antecedent(np.arange(0.0, 5000.0, 1.0), 'distance_to_wp')
        cross_track_error = ctrl.Antecedent(np.arange(-180.0, 180.0, 1.0), 'cross_track_error')
        wave_condition = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'wave_condition')

        # Output variables - direct motor control
        left_motor = ctrl.Consequent(np.arange(0.0, 2.5, 0.1), 'left_motor')
        right_motor = ctrl.Consequent(np.arange(0.0, 2.5, 0.1), 'right_motor')

        # Store variable in a dictionary for easy access
        fuzzy_vars = {
            'distance_to_wp': distance_to_wp,
            'cross_track_error': cross_track_error,
            'wave_condition': wave_condition,
            'left_motor': left_motor,
            'right_motor': right_motor
        }

        return fuzzy_vars
    
    def _define_fuzzy_membership_functions(self):
        """Defining membership functions for all fuzzy variables."""
        # Extract variables ffrom the dictionary
        distance_to_wp = self.fuzzy_variables['distance_to_wp']
        cross_track_error = self.fuzzy_variables['cross_track_error']
        wave_condition = self.fuzzy_variables['wave_condition']
        left_motor = self.fuzzy_variables['left_motor']
        right_motor = self.fuzzy_variables['right_motor']

        # Membership functions for distance to waypoint
        distance_to_wp['close'] = fuzz.trapmf(distance_to_wp.universe, [0.0, 0.0, 3.0, 6.0])
        distance_to_wp['medium'] = fuzz.trimf(distance_to_wp.universe, [3.0, 6.0, 9.0])
        distance_to_wp['far'] = fuzz.trimf(distance_to_wp.universe, [6.0, 9.0, 12.0])
        distance_to_wp['very_far'] = fuzz.trapmf(distance_to_wp.universe, [9.0, 12.0, 5000.0, 5000.0])

        # Membership functions for cross_track_error
        cross_track_error['large_left'] = fuzz.trapmf(cross_track_error.universe, [-180, -180, -1.2, -0.6])
        cross_track_error['small_left'] = fuzz.trimf(cross_track_error.universe, [-1.2, -0.6, 0])
        cross_track_error['center'] = fuzz.trimf(cross_track_error.universe, [-1, 0, 1])
        cross_track_error['small_right'] = fuzz.trimf(cross_track_error.universe, [0, 0.6, 1.2])
        cross_track_error['large_right'] = fuzz.trapmf(cross_track_error.universe, [0.6, 1.2, 180, 180])

        # Membership functions for wave conditions
        wave_condition['going_down'] = fuzz.trimf(wave_condition.universe, [-1, -1, -0.2])
        wave_condition['stable'] = fuzz.trimf(wave_condition.universe, [-0.3, 0, 0.3])
        wave_condition['going_up'] = fuzz.trimf(wave_condition.universe, [0.2, 1, 1])

         # Membership function for left motor
        left_motor['slow'] = fuzz.trapmf(left_motor.universe, [0.0, 0.0, 0.5, 1.0])
        left_motor['moderate'] = fuzz.trimf(left_motor.universe, [0.5, 1.0, 1.5])
        left_motor['fast'] = fuzz.trimf(left_motor.universe, [1.0, 1.5, 2.0])
        left_motor['very_fast'] = fuzz.trapmf(left_motor.universe, [1.5, 2.0, 2.5, 2.5])
        
        # Membership function for right motor
        right_motor['slow'] = fuzz.trapmf(right_motor.universe, [0.0, 0.0, 0.5, 1.0])
        right_motor['moderate'] = fuzz.trimf(right_motor.universe, [0.5, 1.0, 1.5])
        right_motor['fast'] = fuzz.trimf(right_motor.universe, [1.0, 1.5, 2.0])
        right_motor['very_fast'] = fuzz.trapmf(right_motor.universe, [1.5, 2.0, 2.5, 2.5])

    def _set_defuzzification_methods(self):
        """Settin defuzzification methods for output variables."""
        self.fuzzy_variables['left_motor'].defuzzify_method = 'centroid'
        self.fuzzy_variables['right_motor'].defuzzify_method = 'centroid'

    def _create_fuzzy_rules(self):
        """Creating and return the list of fuzzy rules."""
        # Extract variables ffrom the dictionary
        distance = self.fuzzy_variables['distance_to_wp']
        cte = self.fuzzy_variables['cross_track_error']
        wave_condition = self.fuzzy_variables['wave_condition']
        left_motor = self.fuzzy_variables['left_motor']
        right_motor = self.fuzzy_variables['right_motor']

        # Creating rules
        rules = [
            # Center heading rules and straight line navigation
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['stable'], left_motor['very_fast']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['stable'], right_motor['very_fast']),

            # Distance-based speed adjustments
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['stable'], left_motor['fast']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['stable'], right_motor['fast']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['stable'], left_motor['moderate']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['stable'], right_motor['moderate']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['stable'], left_motor['slow']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['stable'], right_motor['slow']),

            # Wave compensation - going up
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_up'], left_motor['very_fast']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_up'], right_motor['very_fast']),

            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_up'], left_motor['very_fast']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_up'], right_motor['very_fast']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_up'], left_motor['very_fast']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_up'], right_motor['very_fast']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_up'], left_motor['fast']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_up'], right_motor['fast']),

            # Wave compensation - going down
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_down'], left_motor['moderate']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_down'], right_motor['moderate']),

            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_down'], left_motor['moderate']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_down'], right_motor['moderate']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_down'], left_motor['moderate']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_down'], right_motor['moderate']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_down'], left_motor['slow']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_down'], right_motor['slow']),

            # Heading correction rules
                # Right turn corrections (reduce right motor speed)
            ctrl.Rule(cte['large_left'], left_motor['very_fast']),
            ctrl.Rule(cte['large_left'], right_motor['slow']),

            ctrl.Rule(cte['small_left'], left_motor['fast']),
            ctrl.Rule(cte['small_left'], right_motor['slow']),

                # Right turn corrections (reduce right motor speed)
            ctrl.Rule(cte['small_right'], left_motor['slow']),
            ctrl.Rule(cte['small_right'], right_motor['fast']),

            ctrl.Rule(cte['large_right'], left_motor['slow']),
            ctrl.Rule(cte['large_right'], right_motor['very_fast'])
            ]

        return rules

    def compute_navigation_parameters(self):
        """Calculate steering output and heading error using fuzzy logic."""       
        # Extract the global coordinates
        target_lat, target_lon, _ = self.waypoints[self.current_wp_index]['global_coordinates']
        prev_lat, prev_lon = self.previous_waypoint

        self.distance_from_prev_wp = self.distance_calculation(prev_lat, prev_lon)
        self.distance_prev_target_wp = self.distance_calculation(prev_lat, prev_lon, target_lat, target_lon)

        # Calculate heading error
        self.compute_heading_error()
        self.cross_track_error = self.calculate_cross_track_error(prev_lat, prev_lon, target_lat, target_lon)

        self.wave_condition = np.clip(self.wave_phase, -1.0, 1.0)

        # Fuzzy input
        self.fuzzy_simulator.input['cross_track_error'] = self.cross_track_error
        self.fuzzy_simulator.input['distance_to_wp'] = self.distance_to_target_wp
        self.fuzzy_simulator.input['wave_condition'] = self.wave_condition
        self.fuzzy_simulator.compute()

        left_motor_output = self.fuzzy_simulator.output['left_motor']
        right_motor_output = self.fuzzy_simulator.output['right_motor']

        self.get_logger().info(
            f"Distance from WP: {self.distance_from_prev_wp:.2f} | " + 
            f"Distance to WP: {self.distance_to_target_wp:.2f}m | " + 
            f"Waypoints distance: {self.distance_prev_target_wp:.2f} | " +
            f"Heading Error: {self.theta_actual_path:.2f}° | " + 
            f"CTE: {self.cross_track_error:.2f}m | " + 
            f"WV: {self.wave_condition:.2f} | " + 
            f"Left motor/Right motor: {left_motor_output:.2f}/{right_motor_output:.2f}")

        self.publish_motor_commands(left_motor_output, right_motor_output)

    def compute_heading_error(self):
        """Compute heading error in degrees using cosine rule, Desired path, actual path and waypoint path"""
        if len(self.waypoints) < 2:
            return
        
        if self.distance_from_prev_wp == 0.0 or self.distance_prev_target_wp == 0.0:
            return
        
        product = (self.distance_from_prev_wp**2 + self.distance_prev_target_wp**2 - self.distance_to_target_wp**2) / (2*self.distance_from_prev_wp*self.distance_prev_target_wp)
        product = max(-1.0, min(1.0, product))
        self.theta_actual_path = math.acos(product)

    def calculate_cross_track_error(self, prev_lat, prev_lon, target_lat, target_lon):
        """ Compute cross-track error from the planned path line to the current position using sin rule."""
        x, y = math.radians(self.global_position.latitude), math.radians(self.global_position.longitude)

        # Determining sign (positive if vehicle is to the right of the path)
        dx, dy = math.radians(target_lat - prev_lat), math.radians(target_lon - prev_lon)

        # Calculating the sign based on which side of the line the vehicle is on
        cross_product = dx * (y - math.radians(prev_lon)) - dy * (x - math.radians(prev_lat))
        sign = 1 if cross_product > 0 else -1
        
        cte = sign * self.distance_from_prev_wp*math.sin(self.theta_actual_path)
        
        return cte

    def publish_motor_commands(self, left_motor_speed, right_motor_speed):
        """Publish direct motor commands using MAVROS ActuatorControl."""
        motor_msg = ActuatorControl()
        motor_msg.header.stamp = self.get_clock().now().to_msg()
        motor_msg.header.frame_id = "base_link"
        
        # Group 0 is typically used for direct actuator control
        motor_msg.group_mix = 0
        motor_msg.controls = [left_motor_speed, right_motor_speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motor_command_publisher.publish(motor_msg)

        # Create and publish status message for the data logging
        status_msg = MyFuzzyOutput()
        status_msg.current_wp_index = self.current_wp_index
        status_msg.distance_to_waypoint = self.distance_to_target_wp
        status_msg.cross_track_error = self.cross_track_error
        status_msg.wave_condition = self.wave_condition
        status_msg.left_motor_speed = left_motor_speed
        status_msg.right_motor_speed = right_motor_speed
        status_msg.waypoints_length = len(self.waypoints)
        status_msg.name = "FLC"
        self.nav_status_publisher.publish(status_msg)
 
    def add_waypoint(self, lat, lon, altitude):
        """Add a waypoint to the mission."""
        if not self.waypoint_push_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Waypoint push service not available')
            return False
        
        # Create a new waypoint
        wp = Waypoint()
        wp.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp.command = 16  # MAV_CMD_NAV_WAYPOINT
        wp.is_current = False
        wp.autocontinue = True
        wp.param1 = 0.0  # Hold time
        wp.param2 = 2.0  # Acceptance radius
        wp.param3 = 0.0  # Pass through
        wp.param4 = float('nan')  # Yaw
        wp.x_lat = lat
        wp.y_long = lon
        wp.z_alt = altitude
        
        # Add to local waypoints list
        self.waypoints.append((lat, lon))
        
        # Create waypoint list for service call
        waypoints = [wp]
        
        # Send waypoint to vehicle
        request = WaypointPush.Request()
        request.start_index = 0
        request.waypoints = waypoints
        
        future = self.waypoint_push_service.call_async(request)
        future.add_done_callback(self.waypoint_push_callback)
        
        return True

    def waypoint_push_callback(self, future):
        """Callback for waypoint push service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Waypoint successfully uploaded')
            else:
                self.get_logger().error('Failed to upload waypoint')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def set_mode(self, mode):
        """Set the flight mode of the vehicle."""
        if not self.set_mode_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Set mode service not available')
            return False
        
        request = SetMode.Request()
        request.custom_mode = mode
        
        future = self.set_mode_service.call_async(request)
        future.add_done_callback(self.set_mode_callback)
        
        return True

    def set_mode_callback(self, future):
        """Callback for set mode service."""
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info('Mode change request sent')
            else:
                self.get_logger().error('Failed to send mode change request')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def arm(self, arm_state):
        """Arm or disarm the vehicle."""
        if not self.arm_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Arming service not available')
            return False
        
        request = CommandBool.Request()
        request.value = arm_state
        
        future = self.arm_service.call_async(request)
        future.add_done_callback(self.arm_callback)
        
        return True

    def arm_callback(self, future):
        """Callback for arm service."""
        try:
            response = future.result()
            if response.success:
                state = "armed" if future.request.value else "disarmed"
                self.get_logger().info(f'Vehicle successfully {state}')
            else:
                self.get_logger().error('Failed to change arm state')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def navigation_timer_callback(self):
        """Timer triggered navigation computation."""
        # Check if we have a mission
        if not self.waypoints:
            self.get_logger().info("No waypoints available")
            return    
        
        # check if the mission is complete
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info("Mission complete - all waypoints reached")
            return 
        
        # Execute navigation if we're in the correct mode and armed
        if self.vehicle_state.armed and self.vehicle_state.mode in ["AUTO.MISSION", "GUIDED", "AUTO"]:
            try:
                self.compute_navigation_parameters()
            except Exception as e:
                self.get_logger().error(f"Navigation computation failed: {e}") 
        else:
            self.get_logger().debug(f"Vehicle not ready - Mode: {self.vehicle_state.mode}, Armed: {self.vehicle_state.armed}")

def main(args=None):
    """Main function to start the node."""
    rclpy.init(args=args)
    node = FuzzyNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Fuzzy Navigation MAVROS node terminated.")
    except Exception as e:
        print(f"Error: {e}")