import time
import math
import logging
import numpy as np
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from map_visualization import Visualizer
from sim_util import sim_car_controls
from vehicle_config import Vehicle_config as conf
from car_state import State, States
from controllers import update_target, AccelerationPIDController, LQGAccelerationController
from sim_util import load_cones_from_lidar,load_cones_from_referee
from logger import log_timing

# Simulation parameters
T = 100.0  # Max simulation time [s]
dt = 0.05  # Time step [s]
Time_zero = time.perf_counter()
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
    path_planner
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
    while T >= curr_time and lastIndex > target_ind:

        # Update state and target
        start_time = time.perf_counter()
        state, target_ind, cx, cy, curve, cones_by_type= update_target(
            client, cx, cy, path_planner, car_position, car_direction, state, target_ind, curve, cones_by_type
        )
        path_track = np.column_stack((np.arange(len(cx)), cx, cy)) #List of XY cords of track
        paths = path_track
        state_update_time = time.perf_counter() - start_time
        print(f"State Update Time: {state_update_time:.4f} seconds")
        log_timing('timing_log.csv', 'State_Update', state_update_time)



        # Control logic
        if hasattr(steering_controller, 'compute_steering'):
            # Compute steering angle
            steering_angle, target_ind = steering_controller.compute_steering(state, path_track, target_ind)
            #compute accelaation using pid
            if isinstance(acceleration_controller,AccelerationPIDController): 
                curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
                acceleration = acceleration_controller.compute_acceleration(state.v, curvature)
                v_log = acceleration_controller.maxspeed

            # Compute acceleration using LQG controller
            elif isinstance(acceleration_controller, LQGAccelerationController):
                curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
                acceleration = acceleration_controller.compute_acceleration(state.v, curvature)
                v_log = acceleration_controller.v_desired


        elif hasattr(steering_controller, 'compute_control'):
            # For controllers like MPC that compute both acceleration and steering
            acceleration, steering_angle = steering_controller.compute_control(state, path_track)
            v_log = state.v  # Use current speed if desired speed is not available
        else:
            raise ValueError("Invalid steering controller type")

        # Send control commands to the simulator
        sim_car_controls(client, -steering_angle, acceleration)

        curr_time += dt
        state_modifier = State(x=state.x, y=-state.y, yaw=state.yaw, v=state.v)
        states.append(
            curr_time,
            state_modifier,
            steering=steering_angle if steering_angle else 0.0,  # Default to 0.0 if undefined
            acceleration=acceleration if acceleration else 0.0,  # Default to 0.0 if undefined
            v_log=v_log if v_log else 0.0  # Default to 0.0 if undefined
        )

        if animate:
            lidar_cones_by_type, _, _ = load_cones_from_lidar(client)
            cones_by_type, car_position, car_direction = load_cones_from_referee(client)
            if lidar_cones_by_type:
                cones_lidar = lidar_cones_by_type[ConeTypes.UNKNOWN]
            else:
                cones_lidar = cones_by_type
            Visualizer.draw_frame(cx, cy, states, cones_by_type, target_ind, state, steering_angle, v_log, cones_lidar)

    if animate:
        Visualizer.show(cx, cy, states)
        Visualizer.plot_cte(dt=dt)
    Visualizer.show(cx, cy, states)  # Existing path visualization
    # New plots
    Visualizer.plot_speed_profile(states)
    Visualizer.plot_path_deviation(cx, cy, states,paths)
    Visualizer.plot_control_inputs(states)
    
