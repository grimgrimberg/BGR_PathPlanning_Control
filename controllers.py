# control.py

import numpy as np
import math
import cvxpy as cp  # For MPC
from simple_pid import PID
from sim_util import load_cones_from_referee, sim_car_state, load_cons_from_lidar
from vehicle_config import Vehicle_config as conf
import control as ctrl
from fsd_path_planning import ConeTypes

# Constants for normalization
MAX_STEER_ANGLE = conf.MAX_STEER  # Maximum steering angle in radians

# PID Controller for Acceleration
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
        self.maxspeed = setpoint

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
    
    def compute_breaking(self, curve, target_ind):
        # Logistic decay
        b = 8 # Sensitivity parameter
        c = 1 # Power parameter
        v_logistic = self.maxspeed / (1 + b * np.abs(curve)**c)
        velocity = v_logistic[target_ind]
        self._update_desired_speed(velocity)
        return velocity


    def _update_desired_speed(self, velocity):
        self.pid.setpoint = velocity

class LQGAccelerationController:
    def __init__(self, dt):
        self.dt = dt

        # Augmented state: [e_v; integral of e_v]
        self.A = np.array([[1, 0],
                           [dt, 1]])
        self.B = np.array([[dt],
                           [0]])
        self.C = np.array([[1, 0]])

        # LQR weighting matrices (tune these parameters)
        q_e_v = 10.0    # Cost for speed error
        q_int_e_v = 100.0  # Cost for integral of speed error
        self.Q = np.diag([q_e_v, q_int_e_v])
        r = 0.001  # Control effort cost
        self.R = np.array([[r]])

        # Solve the discrete-time LQR problem
        self.K, _, _ = ctrl.dlqr(self.A, self.B, self.Q, self.R)
        print(f"LQR Gain K: {self.K}")

        # Kalman filter noise covariances
        process_noise_variance = 0.1
        measurement_noise_variance = 0.5
        self.Q_kf = np.diag([process_noise_variance, process_noise_variance])
        self.R_kf = np.array([[measurement_noise_variance]])

        # Initialize state estimate and covariance
        self.x_hat = np.array([[0.0],  # Initial speed error estimate
                               [0.0]])  # Initial integral error estimate
        self.P = np.eye(2) * 1.0  # Initial estimate covariance

        # Previous control input
        self.a_k_prev = 0.0

        # Initialize desired speed
        self.v_desired = 0.0

    def compute_acceleration(self, current_speed, curvature):
        # Compute desired speed based on curvature
        v_desired = self.compute_desired_speed(curvature)
        self.v_desired = v_desired

        # Measurement (speed error)
        e_v_measurement = current_speed - v_desired

        # Kalman filter update
        self.x_hat, self.P = self.kalman_filter_update(
            self.x_hat, self.P, self.a_k_prev, e_v_measurement)

        # LQR control law
        a_k = -self.K @ self.x_hat

        # Apply acceleration limits
        a_k = np.clip(a_k, conf.MAX_DECEL, conf.MAX_ACCEL)

        # Update previous control input
        self.a_k_prev = a_k.item()

        return a_k.item()



    def kalman_filter_update(self, x_hat, P, a_k_prev, z_k):
        # Predict
        x_pred = self.A @ x_hat + self.B * a_k_prev
        P_pred = self.A @ P @ self.A.T + self.Q_kf

        # Measurement residual
        y = z_k - (self.C @ x_pred)

        # Innovation covariance
        S = self.C @ P_pred @ self.C.T + self.R_kf

        # Kalman gain
        K_kf = P_pred @ self.C.T @ np.linalg.inv(S)

        # Updated state estimate
        x_hat_new = x_pred + K_kf @ y

        # Updated covariance estimate
        P_new = (np.eye(self.A.shape[0]) - K_kf @ self.C) @ P_pred

        return x_hat_new, P_new


    def compute_desired_speed(self, curvature):
        """
        Compute the desired speed based on curvature.

        Args:
            curvature (float): Curvature at the target point.

        Returns:
            float: Desired speed [m/s].
        """
        # Define the relationship between curvature and desired speed
        a = conf.k_expo       # Sensitivity parameter for exponential decay
        
        max_speed = conf.TARGET_SPEED  # Maximum desired speed
        v_desired = max_speed * np.exp(-a * np.abs(curvature))
        return v_desired

# Pure Pursuit Controller
class PurePursuitController:
    def __init__(self, look_ahead_distance=conf.LOOKAHEAD_DISTANCE):
        """
        Pure Pursuit Steering Controller.

        Args:
            look_ahead_distance (float): Look-ahead distance for target point [m].
        """
        self.look_ahead_distance = look_ahead_distance

    def compute_steering(self, state, path, target_ind):
        """
        Compute steering angle using Pure Pursuit algorithm.

        Args:
            state (State): Current state of the vehicle.
            path (numpy.ndarray): Path coordinates [[index, x1, y1], [index, x2, y2], ...].
            target_ind (int): Current target index on the path.

        Returns:
            float: Steering angle [rad].
            int: Updated target index.
        """
        cx = path[:, 1]
        cy = path[:, 2]
        ind = target_ind

        # Search for the target point
        Lf = self.look_ahead_distance
        while ind < len(cx):
            dx = cx[ind] - state.x
            dy = cy[ind] - state.y
            distance = np.hypot(dx, dy)
            if distance >= Lf:
                break
            ind += 1
        if ind >= len(cx):
            ind = len(cx) - 1

        # Calculate the steering angle
        tx = cx[ind]
        ty = cy[ind]
        alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
        alpha = normalize_angle(alpha)  # Normalize angle

        steering = math.atan2(2.0 * conf.WB * math.sin(alpha) / Lf, 1.0)
        steering = np.clip(steering, -conf.MAX_STEER, conf.MAX_STEER)
        return steering, ind
 
# Stanley Controller
class StanleyController:
    def __init__(self, k=1.0, k_soft=1.0, max_steer_rate=np.deg2rad(30), alpha=0.5, lookahead_distance=0.0):
        """
        Improved Stanley Steering Controller.

        Args:
            k (float): Proportional gain for the cross-track error.
            k_soft (float): Softening constant to avoid division by zero.
            max_steer_rate (float): Maximum rate of change of the steering angle [rad/s].
            alpha (float): Smoothing factor for steering command (0 < alpha <= 1).
            lookahead_distance (float): Optional look-ahead distance [m].
        """
        self.k = k
        self.k_soft = k_soft
        self.max_steer_rate = max_steer_rate
        self.alpha = alpha  # For low-pass filtering
        self.lookahead_distance = lookahead_distance
        self.previous_steering = 0.0  # Initialize previous steering angle

    def compute_steering(self, state, path, target_ind, dt):
        """
        Compute steering angle using the improved Stanley control algorithm.

        Args:
            state (State): Current state of the vehicle.
            path (numpy.ndarray): Path coordinates [[index, x1, y1], [index, x2, y2], ...].
            target_ind (int): Current target index on the path.
            dt (float): Time step [s].

        Returns:
            float: Steering angle [rad].
            int: Updated target index.
        """
        cx = path[:, 1]
        cy = path[:, 2]

        # Find the nearest path point (with optional look-ahead)
        dx = cx - state.x
        dy = cy - state.y
        d = np.hypot(dx, dy)
        target_ind = np.argmin(d)

        # Apply look-ahead if specified
        if self.lookahead_distance > 0:
            while target_ind < len(cx) - 1:
                distance = np.hypot(cx[target_ind] - state.x, cy[target_ind] - state.y)
                if distance >= self.lookahead_distance:
                    break
                target_ind += 1

        nearest_x = cx[target_ind]
        nearest_y = cy[target_ind]

        # Path heading at the nearest point
        if target_ind < len(cx) - 1:
            path_dx = cx[target_ind + 1] - cx[target_ind]
            path_dy = cy[target_ind + 1] - cy[target_ind]
        else:
            path_dx = cx[target_ind] - cx[target_ind - 1]
            path_dy = cy[target_ind] - cy[target_ind - 1]

        path_yaw = math.atan2(path_dy, path_dx)

        # Heading error
        theta_e = normalize_angle(path_yaw - state.yaw)

        # Cross-track error using lateral error projection
        dx = nearest_x - state.x
        dy = nearest_y - state.y
        e_ct = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

        # Adaptive gain scheduling based on speed
        adaptive_k = self.k / (state.v + self.k_soft)

        # Steering control law
        steering = theta_e + math.atan2(adaptive_k * e_ct, 1.0)
        steering = normalize_angle(steering)

        # Limit steering rate
        max_delta = self.max_steer_rate * dt
        steering = np.clip(steering, self.previous_steering - max_delta, self.previous_steering + max_delta)

        # Low-pass filter steering angle
        steering = self.alpha * steering + (1 - self.alpha) * self.previous_steering

        # Update previous steering angle
        self.previous_steering = steering

        # Clip to maximum steering angle
        steering = np.clip(steering, -conf.MAX_STEER, conf.MAX_STEER)

        return steering, target_ind
    def __init__(self, k=1.0, k_soft=0.1):
        """
        Stanley Steering Controller.

        Args:
            k (float): Gain for the cross-track error.
            k_soft (float): Softening constant to avoid division by zero.
        """
        self.k = k
        self.k_soft = k_soft

    def compute_steering(self, state, path, target_ind):
        """
        Compute steering angle using Stanley control algorithm.

        Args:
            state (State): Current state of the vehicle.
            path (numpy.ndarray): Path coordinates [[index, x1, y1], [index, x2, y2], ...].
            target_ind (int): Current target index on the path.

        Returns:
            float: Steering angle [rad].
            int: Updated target index.
        """
        cx = path[:, 1]
        cy = path[:, 2]

        # Find the nearest path point
        dx = cx - state.x
        dy = cy - state.y
        d = np.hypot(dx, dy)
        target_ind = np.argmin(d)
        nearest_x = cx[target_ind]
        nearest_y = cy[target_ind]

        # Path heading at the nearest point
        if target_ind < len(cx) - 1:
            path_dx = cx[target_ind + 1] - cx[target_ind]
            path_dy = cy[target_ind + 1] - cy[target_ind]
        else:
            path_dx = cx[target_ind] - cx[target_ind - 1]
            path_dy = cy[target_ind] - cy[target_ind - 1]

        path_yaw = math.atan2(path_dy, path_dx)

        # Heading error
        theta_e = normalize_angle(path_yaw - state.yaw)

        # Cross-track error
        dx = nearest_x - state.x
        dy = nearest_y - state.y
        e_ct = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

        # Steering control law
        steering = theta_e + math.atan2(self.k * e_ct, state.v + self.k_soft)
        steering = normalize_angle(steering)
        steering = np.clip(steering, -conf.MAX_STEER, conf.MAX_STEER)
        return steering, target_ind

class MPCController:
    def __init__(self, N=10, dt=0.1):
        """
        Model Predictive Controller using a linearized vehicle model with fixed speed.

        Args:
            N (int): Prediction horizon.
            dt (float): Time step size [s].
        """
        self.N = N  # Prediction horizon
        self.dt = dt  # Time step
        # Define weights for the cost function
        self.Q = np.diag([1.0, 1.0, 0.5, 0.1])  # State weighting matrix
        self.R = np.diag([0.1, 0.1])  # Control weighting matrix

    def compute_control(self, state, path):
        """
        Compute the optimal control inputs using MPC with a fixed speed.

        Args:
            state (State): Current state of the vehicle.
            path (numpy.ndarray): Path coordinates [[index, x1, y1], [index, x2, y2], ...].

        Returns:
            float: Acceleration [m/s^2].
            float: Steering angle [rad].
        """
        # Extract reference trajectory
        ref_x = path[:, 1]
        ref_y = path[:, 2]

        # Define optimization variables
        x = cp.Variable((self.N + 1, 4))  # States: [x, y, yaw, v]
        u = cp.Variable((self.N, 2))      # Controls: [acceleration, steering]

        # Define constraints and cost
        constraints = []
        cost = 0

        # Initial condition
        constraints += [x[0, :] == [state.x, state.y, state.yaw, state.v]]

        # Precompute cos and sin of the current yaw angle
        cos_theta = np.cos(state.yaw)
        sin_theta = np.sin(state.yaw)

        # Fix the speed to current value
        v_current = state.v

        for k in range(self.N):
            # System dynamics with fixed speed
            constraints += [
                x[k + 1, 0] == x[k, 0] + v_current * cos_theta * self.dt,
                x[k + 1, 1] == x[k, 1] + v_current * sin_theta * self.dt,
                x[k + 1, 2] == x[k, 2] + v_current / conf.WB * u[k, 1] * self.dt,
                x[k + 1, 3] == x[k, 3] + u[k, 0] * self.dt
            ]

            # Control constraints
            constraints += [
                cp.abs(u[k, 0]) <= conf.MAX_ACCEL,
                cp.abs(u[k, 1]) <= conf.MAX_STEER
            ]

            # State constraints (optional)
            constraints += [x[k + 1, 3] >= 0]  # Speed cannot be negative

            # Reference tracking
            if k < len(ref_x):
                ref_state = np.array([ref_x[k], ref_y[k], 0.0, conf.TARGET_SPEED])
            else:
                ref_state = np.array([ref_x[-1], ref_y[-1], 0.0, conf.TARGET_SPEED])

            # Cost function
            cost += cp.quad_form(x[k, :] - ref_state, self.Q) + cp.quad_form(u[k, :], self.R)

        # Terminal cost
        cost += cp.quad_form(x[self.N, :] - ref_state, self.Q)

        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Extract control inputs
            acceleration = u.value[0, 0]
            steering = u.value[0, 1]
        else:
            # If optimization fails, use default values
            acceleration = 0.0
            steering = 0.0
            print("MPC optimization failed. Using zero acceleration and steering.")

        # Clip control inputs to limits
        acceleration = np.clip(acceleration, conf.MAX_DECEL, conf.MAX_ACCEL)
        steering = np.clip(steering, -conf.MAX_STEER, conf.MAX_STEER)

        return acceleration, -steering
    

# Factory function to get steering controller by name
def get_steering_controller(name):
    """
    Factory function to get the steering controller instance based on name.

    Args:
        name (str): Name of the steering controller ('pure_pursuit', 'stanley', 'mpc').

    Returns:
        An instance of the requested steering controller.
    """
    if name == 'pure_pursuit':
        return PurePursuitController(look_ahead_distance=conf.LOOKAHEAD_DISTANCE)
    elif name == 'stanley':
        return StanleyController(k=1.0, k_soft=0.1)
    elif name == 'mpc':
        return MPCController(N=10, dt=0.05)
    else:
        raise ValueError(f"Unknown steering controller name: {name}")

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
        if distance >= conf.LOOKAHEAD_DISTANCE:
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
    # lidar_cones_by_type,car_position, car_direction = load_cons_from_lidar(client)
    # cones_by_type[ConeTypes.UNKNOWN] = lidar_cones_by_type[ConeTypes.UNKNOWN]
    path = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction)
    cx, cy = path[:, 1], -path[:, 2]
    curve = path[:,3]  
    return cx, cy, curve, cones_by_type

def update_path_planner_lidar(client, path_planner, car_position, car_direction):
    """Update the path planner and retrieve new path"""
    # cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    lidar_cones_by_type,car_position, car_direction = load_cons_from_lidar(client)
    # cones_by_type[ConeTypes.UNKNOWN] = lidar_cones_by_type[ConeTypes.UNKNOWN]
    path = path_planner.calculate_path_in_global_frame(lidar_cones_by_type, car_position, car_direction)
    cx, cy = path[:, 1], -path[:, 2]
    curve = path[:,3]  
    return cx, cy, curve, lidar_cones_by_type


def update_target(client, cx, cy, path_planner, car_position, car_direction, state, target_ind, curve, cones_by_type):
    cones_by_type = cones_by_type
    if target_ind > 5:
        # first line is by referee global map
        # cx, cy, curve, cones_by_type = update_path_planner(client, path_planner, car_position, car_direction)
        cx, cy, curve, cones_by_type = update_path_planner_lidar(client, path_planner, car_position, car_direction)
        target_ind = 0
    x, y, yaw, speed = sim_car_state(client)
    state.x, state.y, state.yaw, state.v = x, y, yaw, speed
    target_ind = find_target_point(state, cx, cy, target_ind)
    
    return state, target_ind, cx, cy, curve, cones_by_type
