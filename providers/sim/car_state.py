from typing import Dict
import numpy as np
from providers.sim.sim_util import FSDSClientSingleton
from sub_modules.fsds.client import FSDSClient   # your existing AirSim wrapper
from car_state import State

class SimCarStateProvider:
    def __init__(self):
        self.client = FSDSClientSingleton.instance()

    def start(self):  pass
    def stop(self):   pass

    def read(self) -> Dict[str, np.ndarray]:
        car_state = self.client.getCarState()
        x = car_state.kinematics_estimated.position.x_val
        y = car_state.kinematics_estimated.position.y_val
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi #normalize_yaw
        v = car_state.speed if car_state.speed > 0 else 0
        return {"car_state": State(x=x, y=y, yaw=yaw, v=v)}   # keep as raw object for now
