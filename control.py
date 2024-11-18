# control_airsim.py

import numpy as np
import math
from simple_pid import PID
from sim_util import load_cones_from_referee, sim_car_state
from vehicle_config import Vehicle_config as conf

# Constants for normalization
MAX_STEER_ANGLE = conf.MAX_STEER  # Maximum steering angle in radians

class AccelerationPIDController:
    def __init__(self, kp, ki, kd, setpoint):
        """
        PID Controller for acceleration (throttle control).

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            setpoint (float): Desired speed [m/s].
        """
        self.pid = PID(kp, ki, kd, setpoint)

    def compute_acceleration(self, current_speed):
        """
        Compute acceleration (throttle input) based on current speed.

        Args:
            current_speed (float): Current speed of the vehicle [m/s].

        Returns:
            float: Throttle input [0,1].
        """
        acceleration = self.pid(current_speed)
        acceleration = np.clip(acceleration, conf.MAX_DECEL, conf.MAX_ACCEL)
        return acceleration
    
class SteeringPIDController:
    def __init__(self, kp, ki, kd, setpoint):
        """
        PID Controller for acceleration (throttle control).

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            setpoint (float): Desired speed [m/s].
        """
        self.pid = PID(kp, ki, kd, setpoint)

    def compute_steering(self, heading_error):
        """Compute the steering angle"""

        self.pid.setpoint = 0.0
        di = self.pid(heading_error)
        di = np.clip(di, -conf.MAX_STEER, conf.MAX_STEER)
        # logger.info(f"PID Steering angle: {np.rad2deg(di):.2f}")
        return di

# class PurePursuitController:
#     def __init__(self, look_ahead_distance=conf.LOOKAHEAD_DISTANCE):
#         """
#         Pure Pursuit Steering Controller.

#         Args:
#             look_ahead_distance (float): Look-ahead distance for target point [m].
#         """
#         self.look_ahead_distance = look_ahead_distance

#     def compute_steering(self, state, path, target_ind):
#         """
#         Compute steering angle using Pure Pursuit algorithm.

#         Args:
#             state (State): Current state of the vehicle.
#             path (numpy.ndarray): Path coordinates [[index, x1, y1], [index, x2, y2], ...].
#             target_ind (int): Current target index on the path.

#         Returns:
#             float: Steering angle [rad].
#             int: Updated target index.
#         """
#         cx = path[:, 1]
#         cy = path[:, 2]
#         ind = target_ind

#         # Find the target point at look-ahead distance
#         while ind < len(cx):
#             dx = cx[ind] - state.x
#             dy = cy[ind] - state.y
#             distance = np.hypot(dx, dy)
#             if distance > self.look_ahead_distance:
#                 break
#             ind += 1
#         if ind >= len(cx):
#             ind = len(cx) - 1

#         # Calculate the steering angle
#         tx = cx[ind]
#         ty = cy[ind]
#         alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
#         alpha = normalize_angle(alpha)  # Normalize angle

#         steering = math.atan2(2.0 * conf.WB * math.sin(alpha) / self.look_ahead_distance, 1.0)
#         return steering, ind

# State class
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """
        State of the vehicle.

        Args:
            x (float): Position x [m].
            y (float): Position y [m].
            yaw (float): Heading angle [rad].
            v (float): Speed [m/s].
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

# Utility functions
def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi].

    Args:
        angle (float): Angle [rad].

    Returns:
        float: Normalized angle [rad].
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def find_target_point(state, cx, cy, target_ind):
    """Find the point at lookahead distance"""
    while target_ind < len(cx) - 1:
        distance = np.hypot(cx[target_ind] - state.x, cy[target_ind] - state.y)
        if distance >= conf.LOOKAHEAD_DISTANCE + 2:
            break
        target_ind += 1
    return target_ind

def normalize_angle_180(angle):
    """Normalize angle to [-180, 180] degrees range"""
    angle = math.degrees(angle)  # Convert to degrees
    angle = (angle + 180) % 360 - 180  # Normalize to [-180, 180]
    return math.radians(angle)  # Convert back to radians

def update_path_planner(client, path_planner, car_position, car_direction):
    """Update the path planner and retrieve new path"""
    cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    out = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction,True)
    (
    path,
    sorted_left,
    sorted_right,
    left_cones_with_virtual,
    right_cones_with_virtual,
    left_to_right_match,
    right_to_left_match,
    ) = out
    cx, cy = path[:, 1], -path[:, 2]
    return cx, cy

def update_target(client, cx, cy, path_planner, car_position, car_direction, state, target_ind):
    if target_ind > 10:
        cx, cy = update_path_planner(client, path_planner, car_position, car_direction)
        target_ind = 0
    x, y, yaw, speed = sim_car_state(client)
    state.x, state.y, state.yaw, state.v = x, y, yaw, speed
    # logger.info(f"Updated state: {state}")

    target_ind = find_target_point(state, cx, cy, target_ind)
    target_x, target_y = cx[target_ind], cy[target_ind]

    dx, dy = target_x - state.x, target_y - state.y
    target_yaw = math.atan2(dy, dx)
    heading_error = normalize_angle_180(target_yaw - state.yaw)
    # heading_error = target_yaw - state.yaw

    return state, heading_error, target_ind, cx, cy