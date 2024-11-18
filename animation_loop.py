import time
import math
import logging
import numpy as np 
from fsd_path_planning import PathPlanner, MissionTypes
from map_visualization import Visualizer
from sim_util import sim_car_controls
from vehicle_config import Vehicle_config as conf
from car_state import State ,States
from control import update_target


# Simulation parameters
T = 500000.0  # Max simulation time [s]
dt = 0.05  # Time step [s]

# Visualization settings
animate = True
logger = logging.getLogger('SimLogger')

def animation_main_loop(client, path, car_position, car_direction, cones_by_type, speed_controller, steering_controller, pid_controller=True):
    logger.info("Starting animation main loop")

    cx, cy = path[:, 1], path[:, 2]
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
        
        state, heading_error, target_ind, cx, cy = update_target(client, cx, cy, path_planner, car_position, car_direction, state, target_ind)

        di = steering_controller.compute_steering(heading_error)
        ai = speed_controller.compute_acceleration(state.v)

        sim_car_controls(client, di, ai)

        curr_time += dt
        state_modifier = State(x=state.x, y=-state.y, yaw=state.yaw, v=state.v)
        states.append(curr_time, state_modifier)

        if animate:
            Visualizer.draw_frame(cx, cy, states, cones_by_type, target_ind, state, di)

    if animate:
        Visualizer.show(cx, cy, states)

