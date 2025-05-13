from typing import Dict
import numpy as np
import logging
from vehicle_config import Vehicle_config as conf
from controllers import AccelerationPIDController, PurePursuitController
from providers.sim.sim_util import sim_car_controls    # wrapper for AirSim command
from providers.sim.sim_util import FSDSClientSingleton
from map_visualization import Visualizer
from car_state import State, States
log = logging.getLogger("Controller")

class ControllerSub:
    def __init__(self):
        self.accel_ctrl = AccelerationPIDController(
            kp=conf.kp_accel,
            ki=conf.ki_accel,
            kd=conf.kd_accel,
            setpoint=conf.TARGET_SPEED
        )
        self.steer_ctrl = PurePursuitController(look_ahead_distance=conf.LOOKAHEAD_DISTANCE)
        self.states = States()
    def init(self, providers): pass

    def update(self, data: Dict, dt: float):
        path      = data.get("path")
        car_state = data.get("car_state")
        if path is None or car_state is None:
            return
        state_modifier = State(
            x=car_state.x, 
            y=-car_state.y, 
            yaw=car_state.yaw, 
            v=car_state.v
        )
        self.states.append(0,state_modifier)
        target_ind = data.get("target_ind")
        cones_by_type, car_position, car_direction = data.get("map")
        if data.get("return_intermediate_results"):
            cx, cy = path[0][:, 1], -path[0][:, 2]
            curve = path[0][:, 3]
        else:    
            cx, cy = path[:, 1], -path[:, 2]
            curve = path[:, 3]
        
        
        if path is None or car_state is None:
            return
        
        path_track = np.column_stack((np.arange(len(cx)), cx, cy)) 
        print(f"Car state: x={car_state.x}, y={car_state.y}, yaw={car_state.yaw}, v={car_state.v}")
        
        steering, target_ind   = self.steer_ctrl.compute_steering(car_state, path_track, target_ind)
        curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
        accel      = self.accel_ctrl.compute_acceleration(car_state.v, curvature)
        cones, car_position, car_direction = data.get("cones")
        Visualizer.draw_frame(cx, cy, self.states, cones_by_type, target_ind, car_state, steering, 0, cones)

        sim_car_controls(FSDSClientSingleton.instance(), -steering, accel)