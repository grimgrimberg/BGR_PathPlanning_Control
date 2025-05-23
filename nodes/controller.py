from typing import Dict
import numpy as np
import logging
from vehicle_config import Vehicle_config as conf
from core.controllers import AccelerationPIDController, PurePursuitController
from providers.sim.sim_util import sim_car_controls    # wrapper for AirSim command
from providers.sim.sim_util import FSDSClientSingleton
# from visualization import PlotManager
from core.data.car_state import State, States

# Get logger for this module
log = logging.getLogger("Controller")

class Controller:
    def __init__(self):
        log.info("Initializing Controller with PID and Pure Pursuit controllers")
        self.accel_ctrl = AccelerationPIDController(
            kp=conf.kp_accel,
            ki=conf.ki_accel,
            kd=conf.kd_accel,
            setpoint=conf.TARGET_SPEED
        )
        self.steer_ctrl = PurePursuitController(look_ahead_distance=conf.LOOKAHEAD_DISTANCE)
        self.states = States()
        log.debug(f"Controller initialized with target speed: {conf.TARGET_SPEED}, "
                 f"lookahead distance: {conf.LOOKAHEAD_DISTANCE}")

    def update(self, data: Dict):
        try:
            path = data.get("path")
            cx = data.get("cx")
            cy = data.get("cy")
            curve = data.get("curve")
            car_state = data.get("car_state")
            target_ind = data.get("target_ind")
            
            if any(x is None for x in [path, cx, cy, car_state]):
                log.warning("Missing required data for controller update: "
                          f"path={path is not None}, cx={cx is not None}, "
                          f"cy={cy is not None}, car_state={car_state is not None}")
                return
                    
            state_modifier = State(
                x=car_state.x, 
                y=-car_state.y, 
                yaw=car_state.yaw, 
                v=car_state.v,
                v_linear= car_state.v_linear,
                v_angular= car_state.v_angular,
                a_angular= car_state.a_angular,
                a_linear= car_state.a_linear,
                timestamp=car_state.timestamp
            )
          
            
            path_track = np.column_stack((np.arange(len(cx)), cx, cy)) 
            log.debug(f"Car state - Position: ({car_state.x:.2f}, {car_state.y:.2f}), "
                     f"Yaw: {car_state.yaw:.2f}, Velocity: {car_state.v:.2f}")
            
            # Calculate control inputs
            steering, target_ind = self.steer_ctrl.compute_steering(car_state, path_track, target_ind)
            curvature = curve[target_ind] if target_ind < len(curve) else curve[-1]
            acceleration = self.accel_ctrl.compute_acceleration(car_state.v, curvature)
            
            current_time = data.get("current_time", 0)
            self.states.append(
                current_time,
                state_modifier,
                steering if steering else 0.0,
                acceleration if acceleration else 0.0,
                self.accel_ctrl.maxspeed if self.accel_ctrl.maxspeed else 0.0,
                v_linear= car_state.v_linear,
                v_angular=car_state.v_angular,
                a_linear=car_state.a_angular,
                a_angular=car_state.a_linear,
                timestamp=car_state.timestamp
                )
            data["states"] = self.states # expose to other nodes
            # Log control decisions
            log.debug(f"Control outputs - Steering: {steering:.3f}, Acceleration: {acceleration:.3f}, "
                     f"Target Index: {target_ind}, Curvature: {curvature:.3f}")
            
            # expose to other nodes
            data["v_log"] = self.accel_ctrl.maxspeed
            data["acceleration"] = acceleration
            data["steering"] = steering
            
            # Apply controls and log the action
            log.info(f"Applying controls - Steering: {-steering:.3f}, Acceleration: {acceleration:.3f}")
            sim_car_controls(FSDSClientSingleton.instance(), -steering, 0)
            
        except Exception as e:
            log.error(f"Error in controller update: {str(e)}", exc_info=True)
            raise  # Re-raise the exception after logging
        