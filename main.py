 #main.py
import time
import logging
from logger import init_logger,visualize_timing_data,log_timing
from fsd_path_planning import PathPlanner, MissionTypes
from providers.sim.sim_util import init_client, load_cones_from_referee, load_cones_from_lidar
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
from map_visualization import Visualizer
from Calibration_loop import calibration_loop
import argparse




def main():
    # Initialize the logger
    init_logger()
    logger = logging.getLogger('SimLogger')

    
    # Parse command-line arguments for modes
    
    parser = argparse.ArgumentParser(description="Choose simulation or calibration mode.")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "calibration"],
                        help="Select 'simulation' for standard loop or 'calibration' for test data collection.")
    # python main.py --mode calibration thats what we should run on terminal
    args = parser.parse_args()
    
    return_intermediate_results = False #works also at true\False, works with true, dimensionality issue
    experimental_performance_improvements = True #works also at true\False
    # Initialize client and path planner
    # start_time = time.perf_counter()
    client = init_client()
    if args.mode == "calibration":
        print("Starting calibration mode...")
        calibration_loop(client, duration=50, dt=0.05)  # Run the calibration loop
        return  # Exit after calibration
    else:
        print("Starting simulation mode...")
        
    start_time = time.perf_counter()
    client = init_client()
    path_planner = PathPlanner(MissionTypes.trackdrive,experimental_performance_improvements=True)
    path_init_time = time.perf_counter() - start_time
    print(f"Path Planner Initialization Time: {path_init_time:.4f} seconds")
    log_timing('Initialization', path_init_time)



    # Load initial cones and car state
    start_time = time.perf_counter()
    # cones_by_type, car_position, car_direction = load_cones_from_referee(client)
    lidar_cones_by_type, car_position, car_direction = load_cones_from_lidar(client)
    # lidar_cones_by_type, car_position, car_direction = load_cones_from_lidar1(client) #with clipping

    cones_loading_time = time.perf_counter() - start_time
    print(f"Cones Loading Time: {cones_loading_time:.4f} seconds")
    log_timing('Cone_loading_initial', cones_loading_time)

    logger.info("Initial path calculation...")
    path = path_planner.calculate_path_in_global_frame(lidar_cones_by_type, car_position, car_direction,return_intermediate_results=return_intermediate_results)
    # path = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction,return_intermediate_results=False)


    # Initialize acceleration controller
    acceleration_controller = AccelerationPIDController(
        kp=conf.kp_accel,
        ki=conf.ki_accel,
        kd=conf.kd_accel,
        setpoint=conf.TARGET_SPEED
    )
    # acceleration_controller = LQGAccelerationController(dt=conf.dt) #LQG


    # Select steering controller by name
    steering_controller_name = 'pure_pursuit'  # Options: 'pure_pursuit', 'stanley','mpc'
    # steering_controller_name = 'stanley'  # Options: 'pure_pursuit', 'stanley', 'mpc'
    # steering_controller_nme = 'mpc'  # Options: 'pure_pursuit', 'stanley', 'mpc'

    steering_controller = get_steering_controller(name=steering_controller_name)

    logger.info(f"Using steering controller: {steering_controller_name}")

    # Run the animation and control loop
    start_time = time.perf_counter()

    animation_main_loop(
        client=client,
        path=path,
        car_position=car_position,
        car_direction=car_direction,
        cones_by_type=lidar_cones_by_type,
        # cones_by_type=cones_by_type,
        acceleration_controller=acceleration_controller,
        steering_controller=steering_controller,
        path_planner=path_planner,
        return_intermediate_results=return_intermediate_results,
        experimental_performance_improvements=experimental_performance_improvements
    )
    control_loop_time = time.perf_counter() - start_time
    print(f"Control Loop Execution Time: {control_loop_time:.4f} seconds")
    log_timing('control_loop', control_loop_time)

    



if __name__ == "__main__":
    main()
    print("ive exitied the main and im going to visualize time")
    visualize_timing_data()
    
