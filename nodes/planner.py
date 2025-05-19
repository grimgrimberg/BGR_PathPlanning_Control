from typing import Dict, List
import logging
import numpy as np
from vehicle_config import Vehicle_config as conf
from fsd_path_planning import PathPlanner, MissionTypes

log = logging.getLogger("Planner")

class Planner:
    def __init__(self):
        log.info("Initializing Path Planner for trackdrive mission")
        self.planner = PathPlanner(MissionTypes.trackdrive)
        self.current_path = None
        self.target_ind = 0
        log.debug("Path Planner initialized successfully")

    def update(self, data: Dict, dt: float):
        try:
            cones, car_position, car_direction = data.get("cones")
            
            if cones is None:
                log.warning("No cone data available for path planning")
                return
            
            # Check if new path calculation is needed
            should_recalculate = (self.current_path is None or self.target_ind > 3)
            if should_recalculate:
                log.info(f"Recalculating path (target_ind={self.target_ind})")
                
                # Log the planning inputs
                log.debug(f"Planning inputs - Car position: {car_position}, "
                         f"Car direction: {car_direction}, "
                         f"Number of cones: {sum(len(cone_group) for cone_group in cones)}")
                
                # Calculate new path
                try:
                    self.current_path = self.planner.calculate_path_in_global_frame(
                        cones, car_position, car_direction, return_intermediate_results=False
                    )
                    log.info("Path calculation successful")
                    self.target_ind = 0
                except Exception as e:
                    log.error(f"Path calculation failed: {str(e)}", exc_info=True)
                    return
            
            # Extract path coordinates
            cx, cy = self.current_path[:, 1], -self.current_path[:, 2]
            
            # Update target index
            prev_target_ind = self.target_ind
            while self.target_ind < len(cx) - 1:
                distance = np.hypot(cx[self.target_ind] - data["car_state"].x,
                                 cy[self.target_ind] - data["car_state"].y)
                if distance >= conf.LOOKAHEAD_DISTANCE:
                    break
                self.target_ind += 1
            
            if self.target_ind != prev_target_ind:
                log.debug(f"Target index updated: {prev_target_ind} -> {self.target_ind}")
            
            # Process path data based on intermediate results flag
            if data.get("return_intermediate_results"):
                log.debug("Processing intermediate results")
                cx, cy = self.current_path[0][:, 1], -self.current_path[0][:, 2]
                curve = self.current_path[0][:, 3]
            else:    
                cx, cy = self.current_path[:, 1], -self.current_path[:, 2]
                curve = self.current_path[:, 3]
            
            # Log path statistics
            log.debug(f"Path statistics - Points: {len(cx)}, "
                     f"Target index: {self.target_ind}, "
                     f"Average curvature: {np.mean(curve):.3f}")
            
            # Expose data to other nodes
            data["path"] = self.current_path  
            data["cx"] = cx
            data["cy"] = cy
            data["curve"] = curve
            data["target_ind"] = self.target_ind
            data["return_intermediate_results"] = False
            
        except Exception as e:
            log.error(f"Error in planner update: {str(e)}", exc_info=True)
            raise  # Re-raise the exception after logging