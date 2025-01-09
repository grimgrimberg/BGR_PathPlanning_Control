 #main.py
import time
import logging
from logger import init_logger,visualize_timing_data,log_timing
from fsd_path_planning import PathPlanner, MissionTypes
from sim_util import init_client, load_cones_from_referee, load_cones_from_lidar
from vehicle_config import Vehicle_config as conf
from animation_loop import animation_main_loop
from controllers import (
    AccelerationPIDController,
    get_steering_controller,
    LQGAccelerationController,
)
import matplotlib.pyplot as plt 
from fsd_path_planning import ConeTypes

from Preformance_analysis.simulation_logger import SimulationLogger





def main():
    # Initialize the logger
    init_logger()
    #Initiate SimLogger
    # sim_logger = SimulationLogger("simulation_log.csv")
    logger = logging.getLogger('SimLogger')

    

    # Initialize client and path planner
    start_time = time.perf_counter()
    client = init_client()
    path_planner = PathPlanner(MissionTypes.skidpad)
    path_init_time = time.perf_counter() - start_time
    print(f"Path Planner Initialization Time: {path_init_time:.4f} seconds")
    log_timing('timing_log.csv', 'Initialization', path_init_time)



    # Load initial cones and car state

    start_time = time.perf_counter()
    # lidar_start_time = time.perf_counter
    cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    # lidar_cones_by_type, car_position, car_direction = load_cones_from_lidar(client)
    cones_loading_time = time.perf_counter() - start_time
    # lidar_cones_loading_time = time.perf_counter() - lidar_start_time
    print(f"Cones Loading Time: {cones_loading_time:.4f} seconds")
    log_timing('timing_log.csv', 'Cone_loading_initial', cones_loading_time)



    referee_map = cones_by_type
    logger.info("Initial path calculation...")
    # path = path_planner.calculate_path_in_global_frame(lidar_cones_by_type, car_position, car_direction)
    path = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction,return_intermediate_results=False)


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
    start_time = time.perf_counter()

    animation_main_loop(
        client=client,
        path=path,
        car_position=car_position,
        car_direction=car_direction,
        # cones_by_type=lidar_cones_by_type,
        cones_by_type=cones_by_type,
        acceleration_controller=acceleration_controller,
        steering_controller=steering_controller,
        referee_map=referee_map
    )
    control_loop_time = time.perf_counter() - start_time
    print(f"Control Loop Execution Time: {control_loop_time:.4f} seconds")
    log_timing('timing_log.csv', 'control_loop', control_loop_time)

    



if __name__ == "__main__":
    main()
    print("ive exitied the main and im going to visualize time")
    visualize_timing_data('timing_log.csv')

    # Finalize logging
    # sim_logger.close()

    # Evaluate performance
    # from performance_analysis.evaluate_performance import evaluate_simulation
    # evaluate_simulation("simulation_log.csv")
