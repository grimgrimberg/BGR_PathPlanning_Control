# test_runs_system_id.py

import numpy as np
import csv
from scipy.signal import tf2ss
from sklearn.linear_model import LinearRegression

# Test Profiles
def generate_test_profiles():
    """
    Define test profiles for throttle and steering.

    Returns:
        dict: Dictionary containing throttle and steering profiles.
    """
    return {
        "straight_line": {
            "throttle": {
                "time": [0, 2, 4, 6],
                "values": [0.0, 0.5, 0.8, 0.0],
            },
            "steering": {
                "time": [0, 2, 4, 6],
                "values": [0.0, 0.0, 0.0, 0.0],
            },
        },
        "circular": {
            "throttle": {
                "time": [0, 5, 10, 15],
                "values": [0.03, 0.03, 0.3, 0.3],
            },
            "steering": {
                "time": [0, 5, 10, 15],
                "values": [0.1, 0.2, 0.3, 0.4],
            },
        },
        "figure_eight": {
            "throttle": {
                "time": [0, 5, 10, 15],
                "values": [0.5, 0.6, 0.5, 0.4],
            },
            "steering": {
                "time": [0, 5, 10, 15],
                "values": [0.2, -0.2, 0.2, -0.2],
            },
        },
    }

# Test Driver
def test_course_driver(current_time, profiles):
    """
    Generate throttle and steering inputs for the test course.

    Args:
        current_time (float): Elapsed time in the simulation.
        profiles (dict): Dictionary containing throttle and steering profiles.

    Returns:
        tuple: (throttle, steering) inputs.
    """
    throttle_profile = profiles["throttle"]
    steering_profile = profiles["steering"]

    throttle = np.interp(current_time, throttle_profile["time"], throttle_profile["values"])
    steering = np.interp(current_time, steering_profile["time"], steering_profile["values"])

    return throttle, steering

# Logging Function
def log_data(file, state, input_u):
    """
    Log state and input data to a CSV file.

    Args:
        file (str): File path for logging data.
        state (list): Current state of the car [x, y, yaw, v, yaw_rate, beta].
        input_u (tuple): Control inputs (throttle, steering).
    """
    with open(file, mode='a') as data_file:
        writer = csv.writer(data_file)
        writer.writerow(state + list(input_u))

# System Identification
def identify_dynamics(data_file):
    """
    Perform system identification to estimate state-space matrices.

    Args:
        data_file (str): Path to the CSV file containing logged data.

    Returns:
        tuple: Estimated (A, B, C, D) matrices.
    """
    data = np.loadtxt(data_file, delimiter=",", skiprows=1)
    states = data[:, :6]  # [x, y, yaw, v, yaw_rate, beta]
    inputs = data[:, 6:]  # [throttle, steering]

    X = states[:-1]  # Current states
    U = inputs[:-1]  # Current inputs
    Y = states[1:]   # Next states

    reg = LinearRegression()
    reg.fit(np.hstack([X, U]), Y)
    AB = reg.coef_
    A = AB[:, :X.shape[1]]
    B = AB[:, X.shape[1]:]

    C = np.eye(states.shape[1])  # Outputs = States
    D = np.zeros((states.shape[1], inputs.shape[1]))  # No direct feedthrough

    return A, B, C, D

# Main Function for Test Runs
def main():
    profiles = generate_test_profiles()
    test_data_file = "test_course_data.csv"

    # Initialize CSV file
    with open(test_data_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "yaw", "v", "yaw_rate", "beta", "throttle", "steering"])

    # Simulate test course (replace this with your control loop logic)
    current_time = 0
    dt = 0.05  # Simulation timestep
    for _ in range(1000):  # Simulate for 50 seconds
        throttle, steering = test_course_driver(current_time, profiles["straight_line"])

        # Simulated state update (replace with actual state from simulator)
        state = [0, 0, 0, throttle * 5, 0, 0]  # Example state update

        # Log data
        log_data(test_data_file, state, (throttle, steering))

        current_time += dt

    # Perform system identification
    A, B, C, D = identify_dynamics(test_data_file)

    print("Estimated A Matrix:\n", A)
    print("Estimated B Matrix:\n", B)
    print("Estimated C Matrix:\n", C)
    print("Estimated D Matrix:\n", D)

if __name__ == "__main__":
    main()
