 #main.py

import logging
from logger import init_logger
from fsd_path_planning import PathPlanner, MissionTypes
from sim_util import init_client, load_cones_from_referee
from vehicle_config import Vehicle_config as conf
from animation_loop import animation_main_loop
from control import (
    AccelerationPIDController,
    get_steering_controller,
)

def main():
    # Initialize the logger
    init_logger()
    logger = logging.getLogger('SimLogger')

    # Initialize client and path planner
    client = init_client()
    path_planner = PathPlanner(MissionTypes.trackdrive, experimental_performance_improvements=True)

    # Load initial cones and car state
    cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    logger.info("Initial path calculation...")
    path = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction)

    # Initialize acceleration controller
    acceleration_controller = AccelerationPIDController(
        kp=conf.kp_accel,
        ki=conf.ki_accel,
        kd=conf.kd_accel,
        setpoint=conf.TARGET_SPEED
    )

    # Select steering controller by name
    # steering_controller_name = 'pure_pursuit'  # Options: 'pure_pursuit', 'stanley', 'mpc'
    steering_controller_name = 'pure_pursuit'  # Options: 'pure_pursuit', 'stanley', 'mpc'
    # steering_controller_name = 'mpc'  # Options: 'pure_pursuit', 'stanley', 'mpc'

    steering_controller = get_steering_controller(name=steering_controller_name)

    logger.info(f"Using steering controller: {steering_controller_name}")

    # Run the animation and control loop
    animation_main_loop(
        client=client,
        path=path,
        car_position=car_position,
        car_direction=car_direction,
        cones_by_type=cones_by_type,
        acceleration_controller=acceleration_controller,
        steering_controller=steering_controller,
    )

if __name__ == "__main__":
    main()
