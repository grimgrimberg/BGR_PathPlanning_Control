import time
import numpy as np
from fsd_path_planning import ConeTypes
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle
from vehicle_config import Vehicle_config as conf
import logging
import sub_modules.fsds as fsds # local directory
from sub_modules.fsds.utils import to_eularian_angles
import math
# from visualization import Visualizer
import logging, time, threading
# Initialize logger
logger = logging.getLogger('SimLogger')


class FSDSClientSingleton:
    _instance = None
    _lock     = threading.Lock()

    @classmethod
    def instance(cls):
        """Returns the single FSDSClient, creating it once in a thread-safe way."""
        if cls._instance is None:
            with cls._lock:           # double-checked locking
                if cls._instance is None:
                    logger.info("Initializing FSDS client…")
                    time.sleep(1)
                    client = fsds.FSDSClient()
                    client.confirmConnection()
                    try:
                        client.reset()
                    except Exception:
                        pass
                    client.enableApiControl(True)
                    cls._instance = client
        return cls._instance


# Initialize FSDS Client
def init_client():
    logger.info('Initializing FSDS client...')
    time.sleep(1)
    client = fsds.FSDSClient()
    client.confirmConnection()
    try:
        client.reset()
    except:
        pass
    client.enableApiControl(True)
    return client

def pointgroup_to_cone(group):
    average_x = 0
    average_y = 0
    for point in group:
        average_x += point['x']
        average_y += point['y']
    average_x = average_x / len(group)
    average_y = average_y / len(group)
    return {'x': average_x, 'y': average_y}


def load_cones_from_lidar1(client: fsds.FSDSClient):
    min_cone_distance = 5 # Minimum distance threshold [m]
    max_cone_distance = 40 # Maximum distance threshold [m]

    # Retrieve LiDAR data from the simulator
    lidardata = client.getLidarData(lidar_name='Lidar')

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
    car_position, car_direction, _= get_car_orientation(client)
    car_state = client.getCarState()
    orientation = car_state.kinematics_estimated.orientation
    yaw = to_eularian_angles(orientation)[2]

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
    print(f"lidar detect: {np.array(cones).shape}")
    cones_by_type[ConeTypes.UNKNOWN] = np.array(cones)

    logger.info("Cones by type:")
    for i, cone_group in enumerate(cones_by_type):
        logger.info(f"Type {i}: {len(cone_group)} cones")
    print("plots ocklock")
    # Visualizer.plot_cones(cones_by_type)

    return cones_by_type, car_position, car_direction
    

def get_car_orientation(client: fsds.FSDSClient):
        # Get car state and orientation
    car_state = client.getCarState()
    car_position = np.array([car_state.kinematics_estimated.position.x_val, -car_state.kinematics_estimated.position.y_val])
    orientation = car_state.kinematics_estimated.orientation
    _, _, yaw = to_eularian_angles(orientation)
    car_direction = unit_2d_vector_from_angle(-yaw)
    return car_position, car_direction, yaw

# Load cones from referee
def load_cones_from_referee():
    logger.info("Loading cones from referee...")
    # Referee Colors:
    # 0 = Yellow (Left), 1 = Blue (Right), 2 = OrangeLarge, 3 = OrangeSmall, 4 = Unknown
    referee_state = client.getRefereeState()
    cones_by_colors = {color: [] for color in range(5)}
    cones = referee_state.cones
    initial_position = referee_state.initial_position

    car_position, car_direction, _ = get_car_orientation(client)

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

    logger.info("Cones by type:")
    for i, cones in enumerate(cones_by_type):
        logger.info(f"Type {i}: {len(cones)} cones")

    logger.info(f"Car position: {car_position}")
    logger.info(f"Car direction: {car_direction}")

    return cones_by_type, car_position, car_direction


# Get car state from simulator
def sim_car_state(client):
    car_state = client.getCarState()
    orientation = car_state.kinematics_estimated.orientation
    _, _, yaw = fsds.utils.to_eularian_angles(orientation)
    x = car_state.kinematics_estimated.position.x_val
    y = car_state.kinematics_estimated.position.y_val
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi #normalize_yaw
    v_linear = car_state.kinematics_estimated.linear_velocity
    v_angular = car_state.kinematics_estimated.angular_velocity
    a_linear = car_state.kinematics_estimated.linear_acceleration
    a_angular = car_state.kinematics_estimated.angular_acceleration
    logger.info(f"car_state: {car_state}")
    v = car_state.speed if car_state.speed > 0 else 0
    return x, y, yaw, v, v_linear,v_angular,a_linear,a_angular

# Set car controls in simulator
def sim_car_controls(client, di, ai):
    """
    Sends control commands to the vehicle.

    Args:
        client: AirSim client instance.
        di (float): Steering command [rad].
        ai (float): Acceleration command [m/s^2].
        #kaki
    """
    car_controls = fsds.CarControls()
    car_controls.steering = di / conf.MAX_STEER  # Normalize steering if necessary

    if ai >= 0:
        # Positive acceleration: map to throttle
        throttle_cmd = ai / conf.MAX_ACCEL
        car_controls.throttle = np.clip(throttle_cmd, 0.0, 1.0)
        car_controls.brake = 0.0
    else:
        # Negative acceleration: map to brake
        brake_cmd = -ai / conf.MAX_DECEL  # conf.MAX_DECEL should be negative
        # brake_cmd = ai   # conf.MAX_DECEL should be negative
        car_controls.throttle = 0.0
        car_controls.brake = np.clip(brake_cmd, 0.0, 1.0)

    # Log the control commands
    logger.debug(f"Steering command: {car_controls.steering}")
    logger.debug(f"Throttle command: {car_controls.throttle}")
    logger.debug(f"Brake command: {car_controls.brake}")

    # Send controls to AirSim
    client.setCarControls(car_controls)