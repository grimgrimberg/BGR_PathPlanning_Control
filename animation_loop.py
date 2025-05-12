import time
import math
import logging
import numpy as np
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from map_visualization import Visualizer
from providers.sim.sim_util import sim_car_controls
from vehicle_config import Vehicle_config as conf
from car_state import State, States
from controllers import update_target, AccelerationPIDController, LQGAccelerationController
from providers.sim.sim_util import load_cones_from_lidar,load_cones_from_referee,load_cones_from_lidar1
from logger import log_timing
from scipy.interpolate import splprep, splev

# Simulation parameters
T = 200.0  # Max simulation time [s]
dt = 0.1  # Time step [s]
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
    path_planner,
    return_intermediate_results,
    experimental_performance_improvements
):
    logger.info("Starting animation main loop")
    print(path[0])
    if return_intermediate_results:
        cx, cy = path[0][:, 1], path[0][:, 2]
        curve = path[0][:, 3]
        target_ind = 0
        lastIndex = len(cx) - 1
        print("this is cx", cx)
    else:
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
    Time_end = 0
    path_planner = PathPlanner(MissionTypes.trackdrive,experimental_performance_improvements=experimental_performance_improvements)
    
    full_path = set()

    # Smooth and evenly distribute points
    # tck, u = splprep([x, y], s=0.5)  # Use B-spline smoothing
    # u_new = np.linspace(0, 1, 500)  # Generate evenly spaced points
    # x_smooth, y_smooth = splev(u_new, tck)  # Get smoothed coordinates
    X,Y = [],[]
    while T >= curr_time and lastIndex > target_ind:
        # Update state and target
        start_time = time.perf_counter()
        state, target_ind, cx, cy, curve, cones_by_type,v_linear,v_angular,a_linear,a_angular= update_target(
            client, cx, cy, path_planner, car_position, car_direction, state, target_ind, curve, cones_by_type,return_intermediate_results
        )
        path_track = np.column_stack((np.arange(len(cx)), cx, cy)) #List of XY cords of track
        X.append(path_track[:5,1])
        Y.append(-path_track[:5,2])
        new_points = set(zip(cx, cy))
        full_path.update(new_points)
        state_update_time = time.perf_counter() - start_time
        print(f"State Update Time: {state_update_time:.4f} seconds")
        log_timing('State_Update', state_update_time)



        # Control logic
        if hasattr(steering_controller, 'compute_steering'):
            # Compute steering angle
            steering_angle, target_ind = steering_controller.compute_steering(state, path_track, target_ind)
            # steering_angle, target_ind = steering_controller.compute_steering(state, path_track, target_ind,dt) #version with dt is for new stanley

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
        sim_car_controls(client, -steering_angle, acceleration) #Defult running
        # sim_car_controls(client, -steering_angle, 0)#stanting still for testing

        curr_time += dt
        # state_modifier = State(x=state.x, y=-state.y, yaw=state.yaw, v=state.v)
        state_modifier = State(
        x=state.x, 
        y=-state.y, 
        yaw=state.yaw, 
        v=state.v,
        v_linear=v_linear,             # from your returned variables
        v_angular=v_angular,           
        a_linear=a_linear.x_val,       # extracting from Vector3r
        a_angular=a_angular.z_val      # extracting from Vector3r
    )

        # states.append(
        #     curr_time,
        #     state_modifier,
        #     steering_angle if steering_angle else 0.0,         # steering
        #     acceleration if acceleration else 0.0,             # acceleration
        #     v_log if v_log else 0.0,                             # v_log
        #     v_linear,                                          # v_linear
        #     v_angular,                                         # v_angular
        #     a_linear,                                          # a_linear
        #     a_angular                                          # a_angular

        # )
        states.append(
        curr_time,
        state_modifier,
        steering_angle if steering_angle else 0.0,
        acceleration if acceleration else 0.0,
        v_log if v_log else 0.0,
        v_linear=v_linear,
        v_angular=v_angular,
        a_linear=a_linear,
        a_angular=a_angular
    )

        if animate:
            lidar_cones_by_type, car_position, car_direction = load_cones_from_lidar1(client)
            cones_by_type, _, _ = load_cones_from_referee(client)
            if lidar_cones_by_type:
                cones_lidar = lidar_cones_by_type[ConeTypes.UNKNOWN]
            else:
                cones_lidar = cones_by_type
            Visualizer.draw_frame(cx, cy, states, cones_by_type, target_ind, state, steering_angle, v_log, cones_lidar)
            

    if animate:
        Visualizer.show(cx, cy, states)
        Visualizer.plot_cte(dt=dt)
    # Visualizer.show(cx, cy, states)  # Existing path visualization
    # New plots
    Visualizer.plot_speed_profile(states)
    Visualizer.plot_path_deviation1(cx, cy, states, X,Y,cones_by_type,cones_lidar)
    Visualizer.plot_control_inputs(states)
    # Visualizer.plot_acceleration(a_linear,a_angular,dt)
    # Visualizer.plot_acceleration(States,dt)
    # Visualizer.plot_acceleration(states, dt)
    Visualizer.plot_gg(states, dt)
    Visualizer.plot_all_accelerations(states, dt)



    
