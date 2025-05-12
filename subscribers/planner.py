from typing import Dict, List
import logging
import numpy as np
import vehicle_config as conf
from fsd_path_planning.full_pipeline import PathPlanner

log = logging.getLogger("Planner")

class PlannerSub:
    def __init__(self):
        self.planner = PathPlanner(mission_type="trackdrive")
        self.current_path = None
        self.target_ind = 0

    def init(self, providers): pass

    def update(self, data: Dict, dt: float):
        cones, car_position, car_direction = data.get("cones")
        if cones is None: return
        
        #TODO: check if new Path is needed
        if self.current_path is None or self.target_ind > 3:
            self.current_path = self.planner.calculate_path_in_global_frame(cones, car_position, car_direction, return_intermediate_results=False)
            self.target_ind = 0
        
        cx, cy = self.current_path[:, 1], -self.current_path[:, 2]
        while target_ind < len(cx) - 1:
            distance = np.hypot(cx[target_ind] - data["car_state"].x, cy[target_ind] - data["car_state"].y)
            if distance >= conf.LOOKAHEAD_DISTANCE:
                break
            target_ind += 1
        #TODO: consider keep useing the old path if the new one is bad
        
        data["path"] = self.current_path  # expose to other subscribers
        data["target_ind"] = self.target_ind
        data["return_intermediate_results"] = False