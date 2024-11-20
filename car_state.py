import math
import numpy as np
from vehicle_config import Vehicle_config

class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, beta=0.0, r=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw  # Vehicle heading
        self.v = v  # Velocity
        self.beta = beta  # Slip angle
        self.r = r  # Yaw rate
        self.update_positions()

    def update(self, a, delta):
        # Dynamic bicycle model integration step
        dynamic_model = DynamicBicycleModel()
        x_next = dynamic_model.predict_next_state([self.x, self.y, self.yaw, self.v, self.r, self.beta], [delta, a], dt)

        # Unpack the updated state
        self.x, self.y, self.yaw, self.v, self.r, self.beta = x_next
        self.update_positions()

    def update_positions(self):
        """
        Update the positions of both the rear and front axles of the vehicle based on the current state.
        """
        self.rear_x = self.x - (Vehicle_config.l_f * math.cos(self.yaw))
        self.rear_y = self.y - (Vehicle_config.l_f * math.sin(self.yaw))
        self.front_x = self.x + (Vehicle_config.l_r * math.cos(self.yaw))
        self.front_y = self.y + (Vehicle_config.l_r * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y, use_front=True):
        """
        Calculate the distance between the vehicle and a point.

        Args:
            point_x: x coordinate of the point.
            point_y: y coordinate of the point.
            use_front: Boolean indicating whether to use the front or rear axle for calculation.

        Returns:
            Distance between the vehicle and the point.
        """
        if use_front:
            dx = self.front_x - point_x
            dy = self.front_y - point_y
        else:
            dx = self.rear_x - point_x
            dy = self.rear_y - point_y
        return math.hypot(dx, dy)
class DynamicBicycleModel:
    def __init__(self, m=Vehicle_config.m, I_z=Vehicle_config.I_z, l_f=Vehicle_config.l_f, l_r=Vehicle_config.l_r, c_f=Vehicle_config.c_f, c_r=Vehicle_config.c_r, mu=Vehicle_config.mu):
        """
        Initialize the dynamic bicycle model.

        Args:
            m: Vehicle mass [kg]
            I_z: Yaw moment of inertia [kg*m^2]
            l_f: Distance from the center of mass to the front axle [m]
            l_r: Distance from the center of mass to the rear axle [m]
            c_f: Cornering stiffness of the front tires [N/rad]
            c_r: Cornering stiffness of the rear tires [N/rad]
            mu: Friction coefficient between tires and road
        """
        self.m = m
        self.I_z = I_z
        self.l_f = l_f
        self.l_r = l_r
        self.c_f = c_f
        self.c_r = c_r
        self.mu = mu

    def state_equations(self, x, u, dt):
        """
        Compute the next state of the vehicle given the current state and control inputs.

        Args:
            x: Current state [x_position, y_position, yaw, velocity, yaw_rate, slip_angle]
            u: Control input [steering_angle, acceleration]
            dt: Time step [s]

        Returns:
            x_next: Next state of the vehicle
        """
        x_pos, y_pos, yaw, v, r, beta = x
        delta, a = u

        # Vehicle parameters
        m = self.m
        I_z = self.I_z
        l_f = self.l_f
        l_r = self.l_r
        c_f = self.c_f
        c_r = self.c_r
        mu = self.mu

        # Adding a small epsilon to avoid division by zero
        epsilon = 1e-5
        v = max(v, epsilon)

        # Lateral forces (considering the friction limit using mu)
        F_yf = -c_f * (beta + (l_f * r) / v - delta)
        F_yr = -c_r * (beta - (l_r * r) / v)

        # Limiting the lateral forces by friction (Pacejka-like limitation)
        F_yf = np.clip(F_yf, -mu * m * 9.81, mu * m * 9.81)
        F_yr = np.clip(F_yr, -mu * m * 9.81, mu * m * 9.81)

        # State update equations
        x_pos_next = x_pos + v * np.cos(yaw + beta) * dt
        y_pos_next = y_pos + v * np.sin(yaw + beta) * dt
        yaw_next = yaw + r * dt
        v_next = v + a * dt
        r_next = r + (l_f * F_yf - l_r * F_yr) / I_z * dt
        beta_next = beta + (F_yr / (m * v) - r) * dt

        x_next = [x_pos_next, y_pos_next, yaw_next, v_next, r_next, beta_next]
        return x_next

    def predict_next_state(self, current_state, control_input, dt):
        """
        Predict the next state using the dynamic bicycle model.

        Args:
            current_state: Current state of the vehicle [x, y, yaw, velocity, yaw_rate, slip_angle]
            control_input: Control inputs [steering_angle, acceleration]
            dt: Time step [s]

        Returns:
            next_state: Predicted next state of the vehicle
        """
        next_state = self.state_equations(current_state, control_input, dt)
        return next_state