from typing import Dict
import numpy as np
import logging
import vehicle_config as conf
from controllers import AccelerationPIDController, PurePursuitController
from providers.sim.sim_util import sim_car_controls    # wrapper for AirSim command

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

    def init(self, providers): pass

    def update(self, data: Dict, dt: float):
        path      = data.get("path")
        car_state = data.get("car_state")
        target_ind = data.get("target_ind")
        if data.get("return_intermediate_results"):
            cx, cy = path[0][:, 1], -path[0][:, 2]
            curve = path[0][:, 3]
        else:    
            cx, cy = path[:, 1], -path[:, 2]
            curve = path[:, 3]
        
        
        if path is None or car_state is None:
            return
        
        path_track = np.column_stack((np.arange(len(cx)), cx, cy)) 

        steering   = self.steer_ctrl.compute_steering(car_state, path_track, target_ind)
        curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
        accel      = self.accel_ctrl.compute_acceleration(car_state.speed, curvature)

        sim_car_controls(steer=-steering, throttle=accel)