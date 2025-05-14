# # Calibration_loop.py


# import time
# import csv
# import numpy as np
# from test_runs_system_id import generate_test_profiles, test_course_driver, log_data, identify_dynamics
# from car_state import State
# from map_visualization import Visualizer
# from sim_util import sim_car_controls


# def calibration_loop(client, duration=50, dt=0.05):
#     """
#     Calibration loop for running test profiles and collecting system identification data.

#     Args:
#         client: Simulator client object.
#         duration (float): Total time for calibration in seconds.
#         dt (float): Simulation timestep in seconds.

#     Returns:
#         None
#     """
#     # Initialize test profiles
#     profiles = generate_test_profiles()
#     test_data_file = "test_course_data.csv"

#     # Initialize the CSV file for logging
#     with open(test_data_file, mode='w') as file:
#         writer = csv.writer(file)
#         writer.writerow(["x", "y", "yaw", "v", "yaw_rate", "beta", "throttle", "steering"])

#     # Initialize the car state
#     state = State()
    
#     # Visualization setup (optional)
#     visualizer = Visualizer()

#     # Start calibration loop
#     current_time = 0.0
#     while current_time < duration:
#         time.sleep(dt)

#         # Fetch throttle and steering inputs (choose a profile)
#         throttle, steering = test_course_driver(current_time, profiles["circular"])  # Replace "circular" as needed

#         # Update vehicle state (replace with actual simulation dynamics if available)
#         state.update(throttle, steering)

#         # Log data
#         log_data(test_data_file, [state.x, state.y, state.yaw, state.v, state.r, state.beta], (throttle, steering))

#         # Send control commands to the simulator
#         sim_car_controls(client, steering, throttle)

#         # Visualize current state (optional)
#         # visualizer.draw_frame(
#         #     None, None, None, None, None, state, steering, throttle, None, None
#         # )

#         # Increment time
#         current_time += dt

#     # Perform system identification
#     A, B, C, D = identify_dynamics(test_data_file)

#     print("Estimated A Matrix:\n", A)
#     print("Estimated B Matrix:\n", B)
#     print("Estimated C Matrix:\n", C)
#     print("Estimated D Matrix:\n", D)


# if __name__ == "__main__":
#     # Replace `client` with your actual simulator client initialization
#     client = None  # Example placeholder
#     calibration_loop(client)
import time
import csv
import numpy as np
from old.test_runs_system_id import generate_test_profiles, test_course_driver, log_data, identify_dynamics
from core.visualization import Visualizer
from providers.sim.sim_util import sim_car_controls


def calibration_loop(client, duration=50, dt=0.05):
    """
    Calibration loop for running test profiles and collecting system identification data.

    Args:
        client: Simulator client object.
        duration (float): Total time for calibration in seconds.
        dt (float): Simulation timestep in seconds.

    Returns:
        None
    """
    # Initialize test profiles
    profiles = generate_test_profiles()
    test_data_file = "test_course_data.csv"

    # Initialize the CSV file for logging
    with open(test_data_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "x", "y", "z", "vx", "vy", "vz", 
            "ax", "ay", "az", "yaw", "pitch", "roll", 
            "angular_vx", "angular_vy", "angular_vz", 
            "angular_ax", "angular_ay", "angular_az", 
            "throttle", "steering"
        ])

    # Visualization setup (optional)
    visualizer = Visualizer()

    # Start calibration loop
    current_time = 0.0
    while current_time < duration:
        time.sleep(dt)

        # Fetch throttle and steering inputs
        throttle, steering = test_course_driver(current_time, profiles["circular"])  # Replace "circular" as needed

        # Fetch the car's current state from the simulator
        car_state = client.getCarState()
        kinematics = car_state.kinematics_estimated

        # Extract relevant data
        timestamp = car_state.timestamp
        position = kinematics.position
        orientation = kinematics.orientation
        linear_velocity = kinematics.linear_velocity
        angular_velocity = kinematics.angular_velocity
        linear_acceleration = kinematics.linear_acceleration
        angular_acceleration = kinematics.angular_acceleration

        # Convert orientation (Quaternion) to Euler angles (yaw, pitch, roll)
        yaw, pitch, roll = fsds.utils.to_eularian_angles(orientation)

        # Log data
        with open(test_data_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, position.x_val, position.y_val, position.z_val, 
                linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val, 
                linear_acceleration.x_val, linear_acceleration.y_val, linear_acceleration.z_val, 
                yaw, pitch, roll, 
                angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val, 
                angular_acceleration.x_val, angular_acceleration.y_val, angular_acceleration.z_val, 
                throttle, steering
            ])

        # Send control commands to the simulator
        sim_car_controls(client, steering, throttle)

        # Visualize current state (optional)
        visualizer.draw_frame(
            None, None, None, None, None, None, steering, throttle, None, None
        )

        # Increment time
        current_time += dt

    # Perform system identification
    A, B, C, D = identify_dynamics(test_data_file)

    print("Estimated A Matrix:\n", A)
    print("Estimated B Matrix:\n", B)
    print("Estimated C Matrix:\n", C)
    print("Estimated D Matrix:\n", D)


if __name__ == "__main__":
    # Replace `client` with your actual simulator client initialization
    client = None  # Example placeholder
    calibration_loop(client)
