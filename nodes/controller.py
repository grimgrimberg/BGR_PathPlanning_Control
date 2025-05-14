from typing import Dict
import numpy as np
import logging
from vehicle_config import Vehicle_config as conf
from core.controllers import AccelerationPIDController, PurePursuitController
from providers.sim.sim_util import sim_car_controls    # wrapper for AirSim command
from providers.sim.sim_util import FSDSClientSingleton
# from visualization import PlotManager
from core.data.car_state import State, States
log = logging.getLogger("Controller")

class Controller:
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
        cx       = data.get("cx")
        cy       = data.get("cy")
        curve    = data.get("curve")
        car_state = data.get("car_state")
        target_ind = data.get("target_ind")
        if path is None or cx is None or cy is None or car_state is None:
            return
                
        state_modifier = State(
            x=car_state.x, 
            y=-car_state.y, 
            yaw=car_state.yaw, 
            v=car_state.v
        )
        self.states.append(0,state_modifier)
        data["states"] = self.states # expose to other nodes
        
        path_track = np.column_stack((np.arange(len(cx)), cx, cy)) 
        print(f"Car state: x={car_state.x}, y={car_state.y}, yaw={car_state.yaw}, v={car_state.v}")
        
        steering, target_ind   = self.steer_ctrl.compute_steering(car_state, path_track, target_ind)
        curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
        acceleration      = self.accel_ctrl.compute_acceleration(car_state.v, curvature)
        # expose to other nodes
        data["v_log"] = self.accel_ctrl.maxspeed
        data["acceleration"] = acceleration
        data["steering"] = steering
        
        # PlotManager.draw_frame(cx, cy, self.states, data.get("cones_map"), target_ind, car_state, steering, 0, data.get("cones_lidar"))

        sim_car_controls(FSDSClientSingleton.instance(), -steering, acceleration)
        