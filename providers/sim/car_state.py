from typing import Dict
import numpy as np
from providers.sim.sim_util import FSDSClientSingleton
from sub_modules.fsds.client import FSDSClient   # your existing AirSim wrapper
from sub_modules.fsds.utils import to_eularian_angles
from core.data.car_state import State
import logging

log = logging.getLogger("SimLogger")
class SimCarStateProvider:
    def __init__(self):
        self.client = FSDSClientSingleton.instance()
        log.debug(f"Provider {self.__class__.__name__} initialized successfully")
        

    def start(self):  pass
    def stop(self):   pass

    def read(self) -> Dict[str, np.ndarray]:
        car_state = self.client.getCarState()
        orientation = car_state.kinematics_estimated.orientation
        _, _, yaw = to_eularian_angles(orientation)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi #normalize_yaw
        x = car_state.kinematics_estimated.position.x_val
        y = car_state.kinematics_estimated.position.y_val
        v = car_state.speed if car_state.speed > 0 else 0
        return {"car_state": State(x=x, y=y, yaw=yaw, v=v)}   # keep as raw object for now
