from typing import Dict
import math
import logging
import numpy as np
from providers.sim.sim_util import FSDSClientSingleton, get_car_orientation, pointgroup_to_cone
from sub_modules.fsds.utils import to_eularian_angles
from fsd_path_planning import ConeTypes
from .sim_util import pointgroup_to_cone


log = logging.getLogger("SimLogger")


class SimConeProvider:
    def __init__(self):
        self.client = FSDSClientSingleton.instance()
        log.debug(f"Provider {self.__class__.__name__} initialized successfully")
    
    def start(self):     pass
    def stop(self):      pass

    def read(self) -> Dict[str, np.ndarray]:
        min_cone_distance = 0 # Minimum distance threshold [m]
        max_cone_distance = 40 # Maximum distance threshold [m]

        # Retrieve LiDAR data from the simulator
        lidardata = self.client.getLidarData(lidar_name='Lidar')

        # Check if sufficient points are available
        if len(lidardata.point_cloud) < 3:
            return None, None, None

        # Convert LiDAR data into a numpy array of xyz points
        points = np.array(lidardata.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Compute distances from the LiDAR origin to each point
        distances = np.linalg.norm(points[:, :2], axis=1)

        # Filter points based on the specified range limits
        valid_indices = np.logical_and(distances >= min_cone_distance, distances <= max_cone_distance)
        points = points[valid_indices]

        # Obtain car's orientation and position
        car_position, car_direction, yaw = get_car_orientation(self.client)

        # Group points to identify cones
        current_group, cones = [], []
        for i in range(1, len(points)):
            distance_to_last_point = math.dist([points[i][0], points[i][1]], [points[i-1][0], points[i-1][1]])
            
            if distance_to_last_point < 0.1:
                current_group.append({'x': points[i][0], 'y': points[i][1]})
            else:
                if current_group:
                    cone = pointgroup_to_cone(current_group)
                    old_x, old_y = cone['x'] + 1.3, -cone['y']  # Adjust for LiDAR position relative to the car center
                    # Rotate points to global reference frame
                    cone['x'] = np.cos(-yaw) * old_x - np.sin(-yaw) * old_y + car_position[0]
                    cone['y'] = np.sin(-yaw) * old_x + np.cos(-yaw) * old_y + car_position[1]

                    cones.append(np.array([cone['x'], cone['y']]))
                current_group = []

        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
        log.info(f"lidar detect: {np.array(cones).shape}")
        cones_by_type[ConeTypes.UNKNOWN] = np.array(cones)
        cones_lidar = np.array(cones)
        return {
            'cones':(cones_by_type, car_position, car_direction),
            'cones_lidar':cones_lidar    
            }