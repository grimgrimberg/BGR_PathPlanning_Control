import sys
sys.path.insert(0,"/home/bgr/Desktop/roy_ws/first_autonomous_experiment/BGR-PM/Sensors/Lidar/src")
from algo.detection.python.full_detection_block import LidarModule

from typing import Dict
import math
import logging
import numpy as np
from providers.sim.sim_util import FSDSClientSingleton, get_car_orientation, pointgroup_to_cone

from fsd_path_planning import ConeTypes



log = logging.getLogger("SimLogger")


class LidarConeProvider:
    def __init__(self):
        self.lidar_mod = LidarModule()
        log.debug(f"Provider {self.__class__.__name__} initialized successfully")
    
    def start(self):     pass
    def stop(self):      pass

    def read(self) -> Dict[str, np.ndarray]:

        detections = self.lidar_mod.scan_detect()
        lidar_cones = np.array([detections[i][:2] for i in sorted(detections.keys())])
        
        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]

        cones_by_type[ConeTypes.UNKNOWN] = np.array(lidar_cones)/100
        cones_lidar = np.array(lidar_cones)
        return {
            'cones':(cones_by_type, 0, 0),
            'cones_lidar':cones_lidar    
            }