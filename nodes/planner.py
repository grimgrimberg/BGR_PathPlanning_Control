from typing import Dict, List
import logging
import numpy as np
from vehicle_config import Vehicle_config as conf
from fsd_path_planning import PathPlanner, MissionTypes

log = logging.getLogger("Planner")

class Planner:
    def __init__(self):
        self.planner = PathPlanner(MissionTypes.trackdrive)
        self.current_path = None
        self.target_ind = 0

    def init(self, providers): pass

    def update(self, data: Dict, dt: float):
        cones, car_position, car_direction = data.get("cones")
        # cones, car_position, car_direction = data.get("map")

        if cones is None: return
        
        #TODO: check if new Path is needed
        if self.current_path is None or self.target_ind > 3:
            print("Recalculating path!!!!! \n\n\n\n target_ind = ", self.target_ind)
            self.current_path = self.planner.calculate_path_in_global_frame(cones, car_position, car_direction, return_intermediate_results=False)
            self.target_ind = 0
        
        cx, cy = self.current_path[:, 1], -self.current_path[:, 2]
        while self.target_ind < len(cx) - 1:
            distance = np.hypot(cx[self.target_ind] - data["car_state"].x, cy[self.target_ind] - data["car_state"].y)
            if distance >= conf.LOOKAHEAD_DISTANCE:
                break
            self.target_ind += 1
        #TODO: consider keep useing the old path if the new one is bad
        if data.get("return_intermediate_results"):
            cx, cy = self.current_path[0][:, 1], -self.current_path[0][:, 2]
            curve = self.current_path[0][:, 3]
        else:    
            cx, cy = self.current_path[:, 1], -self.current_path[:, 2]
            curve = self.current_path[:, 3]

        # expose to other nodes
        data["path"] = self.current_path  
        data["cx"] = cx
        data["cy"] = cy
        data["curve"] = curve
        data["target_ind"] = self.target_ind
        data["return_intermediate_results"] = False