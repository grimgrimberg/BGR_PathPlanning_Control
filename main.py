 #main.py

import logging
from logger import init_logger
from fsd_path_planning import PathPlanner, MissionTypes
from sim_util import init_client, load_cones_from_referee, load_cons_from_lidar
from vehicle_config import Vehicle_config as conf
from animation_loop import animation_main_loop
from controllers import (
    AccelerationPIDController,
    get_steering_controller,
    LQGAccelerationController,
)
import matplotlib.pyplot as plt 
from fsd_path_planning import ConeTypes

def main():
    # Initialize the logger
    init_logger()
    logger = logging.getLogger('SimLogger')

    # Initialize client and path planner
    client = init_client()
    path_planner = PathPlanner(MissionTypes.trackdrive)

    # Load initial cones and car state
    cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    lidar_cones_by_type, car_position, car_direction = load_cons_from_lidar(client)
    referee_map = cones_by_type
# Plotting the cones
    # plt.figure(1)
    # plt.scatter(cones_by_type[ConeTypes.LEFT][:, 0], cones_by_type[ConeTypes.LEFT][:, 1],
    #             color='yellow', label='Left Cones')
    # plt.scatter(cones_by_type[ConeTypes.RIGHT][:, 0], cones_by_type[ConeTypes.RIGHT][:, 1],
    #             color='blue', label='Right Cones')
    # plt.legend()
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Cone Positions by Type (Left and Right)')

    # # Plotting UNKNOWN cones in a separate figure
    # # plt.figure(2)
    # # plt.clf()
    # # cones_range_cutoff = 40
    # # # plt.axis([-cones_range_cutoff, cones_range_cutoff, -2, cones_range_cutoff])
    # plt.scatter(lidar_cones_by_type[ConeTypes.UNKNOWN][:, 0], lidar_cones_by_type[ConeTypes.UNKNOWN][:, 1],
    #             color='gray', label='Unknown Lidar Cones')
    # plt.legend()
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Cone Positions by Type')

    # plt.show()
    # exit()
    logger.info("Initial path calculation...")
    path = path_planner.calculate_path_in_global_frame(lidar_cones_by_type, car_position, car_direction)

    # Initialize acceleration controller
    acceleration_controller = AccelerationPIDController(
        kp=conf.kp_accel,
        ki=conf.ki_accel,
        kd=conf.kd_accel,
        setpoint=conf.TARGET_SPEED
    )
    # acceleration_controller = LQGAccelerationController(dt=conf.dt) #LQG


    # Select steering controller by name
    # steering_controller_name = 'pure_pursuit'  # Options: 'pure_pursuit', 'stanley', 'mpc'
    steering_controller_name = 'stanley'  # Options: 'pure_pursuit', 'stanley', 'mpc'
    # steering_controller_nme = 'mpc'  # Options: 'pure_pursuit', 'stanley', 'mpc'

    steering_controller = get_steering_controller(name=steering_controller_name)

    logger.info(f"Using steering controller: {steering_controller_name}")
    # cones_by_type[ConeTypes.UNKNOWN] = lidar_cones_by_type[ConeTypes.UNKNOWN]
    # Run the animation and control loop
    animation_main_loop(
        client=client,
        path=path,
        car_position=car_position,
        car_direction=car_direction,
        cones_by_type=lidar_cones_by_type,
        acceleration_controller=acceleration_controller,
        steering_controller=steering_controller,
        referee_map=referee_map
    )

if __name__ == "__main__":
    main()
