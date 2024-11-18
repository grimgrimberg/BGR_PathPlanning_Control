import logging
from logger import init_logger
from fsd_path_planning import PathPlanner, MissionTypes
from sim_util import init_client, load_cones_from_referee
from vehicle_config import Vehicle_config as conf
from animation_loop import animation_main_loop
from simple_pid import PID
from control import SteeringPIDController, AccelerationPIDController

# sys.path.insert(0, '/home/roy/Desktop/SimuClone/BGRacing_Path_Planner_Control/pp_bm_test')
# from falcon_serial import VCU

def main():
    init_logger()
    logger = logging.getLogger('SimLogger')

    client = init_client()
    path_planner = PathPlanner(MissionTypes.trackdrive, experimental_performance_improvements=True)
    cones_by_type, car_position, car_direction = load_cones_from_referee(client)

    logger.info("inital path calc...")
    path = path_planner.calculate_path_in_global_frame(cones_by_type, car_position, car_direction)
    

    # logger.info("initalze controllers ..")
    # speed_controller = PID(conf.kp, conf.ki, conf.kd, conf.target_speed)
    # steering_controller = PID(1.0, 0.0, 0.2, 0.0)
    speed_controller = AccelerationPIDController(conf.kp, conf.ki, conf.kd, conf.target_speed)
    steering_controller = SteeringPIDController(1.5, 0.5 , 0.5, 00)

    animation_main_loop(client, path, car_position, car_direction, cones_by_type, speed_controller, steering_controller)

if __name__ == "__main__":
    main()
