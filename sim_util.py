import time
import numpy as np
from fsd_path_planning import ConeTypes
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle
from vehicle_config import Vehicle_config as conf
import logging
import fsds # local directory

# Initialize logger
logger = logging.getLogger('SimLogger')


# Initialize FSDS Client
def init_client():
    logger.info('Initializing FSDS client...')
    time.sleep(1)
    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(True)
    return client

# Load cones from referee
def load_cones_from_referee(client):
    logger.info("Loading cones from referee...")

    # Referee Colors:
    # 0 = Yellow (Left), 1 = Blue (Right), 2 = OrangeLarge, 3 = OrangeSmall, 4 = Unknown
    referee_state = client.getRefereeState()
    cones_by_colors = {color: [] for color in range(5)}
    cones = referee_state.cones
    initial_position = referee_state.initial_position

    # Get car state and orientation
    car_state = client.getCarState()
    car_position = np.array([car_state.kinematics_estimated.position.x_val, -car_state.kinematics_estimated.position.y_val])
    orientation = car_state.kinematics_estimated.orientation
    _, _, yaw = fsds.utils.to_eularian_angles(orientation)
    yaw = -yaw  # Convert to ENU
    car_direction = unit_2d_vector_from_angle(yaw)

    # Create yaw rotation matrix
    yaw = np.radians(yaw)
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    for cone in cones:
        color = cone['color']
        if color in cones_by_colors:
            x = cone['x'] / 100 - (initial_position.x / 100)
            y = cone['y'] / 100 - (initial_position.y / 100)

            # Rotate cone position to car frame
            cone_position = np.array([x, y])
            #disable rotation on cones seems not relevant
            # rotated_cone_position = rotation_matrix @ cone_position
            # cones_by_colors[color].append(rotated_cone_position)
            cones_by_colors[color].append(cone_position)


    cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
    cones_by_type[ConeTypes.LEFT] = np.array(cones_by_colors[0])  # Yellow cones (Left)
    cones_by_type[ConeTypes.RIGHT] = np.array(cones_by_colors[1])  # Blue cones (Right)
    # cones_by_type[ConeTypes.ORANGE_BIG] = np.array(cones_by_colors[2])  # Large Orange cones
    # cones_by_type[ConeTypes.ORANGE_SMALL] = np.array(cones_by_colors[3])  # Small Orange cones

    logger.info("Cones by type:")
    for i, cones in enumerate(cones_by_type):
        logger.info(f"Type {i}: {len(cones)} cones")

    logger.info(f"Car position: {car_position}")
    logger.info(f"Car direction: {car_direction}")

    return cones_by_type, car_position, car_direction

# Normalize yaw angle to [-pi, pi]
def normalize_yaw(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi

# Get car state from simulator
def sim_car_state(client):
    car_state = client.getCarState()
    orientation = car_state.kinematics_estimated.orientation
    _, _, yaw = fsds.utils.to_eularian_angles(orientation)
    x = car_state.kinematics_estimated.position.x_val
    y = car_state.kinematics_estimated.position.y_val
    yaw = normalize_yaw(yaw)
    return x, y, yaw, car_state.speed

# Set car controls in simulator
def sim_car_controls(client, di, ai):
    car_controls = fsds.CarControls()
    car_controls.steering = di
    car_controls.throttle = ai / conf.MAX_ACCEL
    car_controls.brake = 0.0

    # Ensure controls are within valid range
    car_controls.throttle = np.clip(car_controls.throttle, 0.0, 1.0)
    car_controls.brake = np.clip(car_controls.brake, 0.0, 1.0)

    logger.info(f"Setting simulator controls: {car_controls}")
    client.setCarControls(car_controls)
