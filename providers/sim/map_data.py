from typing import Dict
import numpy as np
from providers.sim.sim_util import FSDSClientSingleton, get_car_orientation
from fsd_path_planning import ConeTypes

class SimMapDataProvider:
    def __init__(self):
        self.client = FSDSClientSingleton.instance()

    def start(self):  pass
    def stop(self):   pass

    def read(self) -> Dict[str, np.ndarray]:
        # Referee Colors:
        # 0 = Yellow (Left), 1 = Blue (Right), 2 = OrangeLarge, 3 = OrangeSmall, 4 = Unknown
        referee_state = self.client.getRefereeState()
        cones_by_colors = {color: [] for color in range(5)}
        cones = referee_state.cones
        initial_position = referee_state.initial_position

        car_position, car_direction, _ = get_car_orientation(self.client)

        for cone in cones:
            color = cone['color']
            if color in cones_by_colors:
                x = cone['x'] / 100 - (initial_position.x / 100)
                y = cone['y'] / 100 - (initial_position.y / 100)

                # Rotate cone position to car frame
                cone_position = np.array([x, y])
                cones_by_colors[color].append(cone_position)

        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
        cones_by_type[ConeTypes.LEFT] = np.array(cones_by_colors[0])  # Yellow cones (Left)
        cones_by_type[ConeTypes.RIGHT] = np.array(cones_by_colors[1])  # Blue cones (Right)
        cones_by_type[ConeTypes.ORANGE_BIG] = np.array(cones_by_colors[2])  # Large Orange cones
        cones_by_type[ConeTypes.ORANGE_SMALL] = np.array(cones_by_colors[3])  # Small Orange cones

        return {'map':(cones_by_type, car_position, car_direction)}
