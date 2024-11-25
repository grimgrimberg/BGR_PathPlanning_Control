from dataclasses import dataclass
import numpy as np

@dataclass
class Vehicle_config():
    # Vehicle parameters
    m = 250  # Vehicle mass [kg]
    I_z = 1700  # Moment of inertia [kg*m^2]
    l_f = 0.835  # Distance from the center of mass to the front axle [m]
    l_r = 0.705  # Distance from the center of mass to the rear axle [m]
    c_f = 16000  # Cornering stiffness front [N/rad]
    c_r = 17000  # Cornering stiffness rear [N/rad]
    mu = 1.0  # Coefficient of friction
    target_speed = 15.0 / 3.6  # Target speed [m/s]
    WB = l_f + l_r 

    # PID Controller parameters
    kp = 2.0
    ki = 0.5
    kd = 0.2


    # Define limits
    MAX_ACCEL = 1.5  # [m/s^2]
    MAX_DECEL = -2.0  # [m/s^2]
    MAX_SPEED = 20.0 / 3.6  # [m/s]
    MAX_STEER = np.deg2rad(30)  # [rad]
    # Lookahead distance for pure pursuit
    LOOKAHEAD_DISTANCE = 5 # [m]