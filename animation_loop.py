import time
import math
import logging
import numpy as np
from fsd_path_planning import PathPlanner, MissionTypes
from map_visualization import Visualizer
from sim_util import sim_car_controls
from vehicle_config import Vehicle_config as conf
from car_state import State, States
from controllers import update_target, AccelerationPIDController, LQGAccelerationController

# Simulation parameters
T = 500000.0  # Max simulation time [s]
dt = 0.05  # Time step [s]

# Visualization settings
animate = True
logger = logging.getLogger('SimLogger')

def animation_main_loop(
    client,
    path,
    car_position,
    car_direction,
    cones_by_type,
    acceleration_controller,
    steering_controller,
):
    logger.info("Starting animation main loop")

    cx, cy = path[:, 1], path[:, 2]
    curve = path[:, 3]
    target_ind = 0
    lastIndex = len(cx) - 1

    # Initial state
    state = State(
        x=car_position[0],
        y=car_position[1],
        yaw=np.arctan2(car_direction[1], car_direction[0]),
        v=0.0
    )
    logger.info(f"Initial state: {state}")

    curr_time = 0.0
    states = States()
    states.append(curr_time, state)

    path_planner = PathPlanner(MissionTypes.trackdrive)

    while T >= curr_time and lastIndex > target_ind:
        time.sleep(dt)

        # Update state and target
        state, target_ind, cx, cy, curve = update_target(
            client, cx, cy, path_planner, car_position, car_direction, state, target_ind, curve
        )
        path = np.column_stack((np.arange(len(cx)), cx, cy))

        # Control logic
        if hasattr(steering_controller, 'compute_steering'):
            # Compute steering angle
            steering_angle, target_ind = steering_controller.compute_steering(state, path, target_ind)
            #compute accelaation using pid
            if isinstance(acceleration_controller,AccelerationPIDController): 
                acceleration = acceleration_controller.compute_acceleration(state.v)
                v_log = state.v
            # Compute acceleration using LQG controller
            elif isinstance(acceleration_controller, LQGAccelerationController):
                curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
                acceleration = acceleration_controller.compute_acceleration(state.v, curvature)
                v_log = acceleration_controller.v_desired

        elif hasattr(steering_controller, 'compute_control'):
            # For controllers like MPC that compute both acceleration and steering
            acceleration, steering_angle = steering_controller.compute_control(state, path)
            v_log = state.v  # Use current speed if desired speed is not available
        else:
            raise ValueError("Invalid steering controller type")

        # Send control commands to the simulator
        sim_car_controls(client, -steering_angle, acceleration)

        curr_time += dt
        state_modifier = State(x=state.x, y=-state.y, yaw=state.yaw, v=state.v)
        states.append(curr_time, state_modifier)

        if animate:
            Visualizer.draw_frame(cx, cy, states, cones_by_type, target_ind, state, steering_angle, v_log)

    if animate:
        Visualizer.show(cx, cy, states)
