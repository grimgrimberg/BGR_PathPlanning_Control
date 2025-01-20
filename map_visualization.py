import math
import numpy as np
import matplotlib.pyplot as plt
from fsd_path_planning import ConeTypes
from vehicle_config import Vehicle_config as conf
# from animation_loop import v_log

class Visualizer:
    @staticmethod
    def plot_car(x, y, yaw, steer=0.0, truckcolor="-k"):
        """
        Plot a car at a specified position, orientation, and steering angle.
        """
        LENGTH = 4.5  # [m]
        WIDTH = 2.0  # [m]
        BACKTOWHEEL = 1.0  # [m]
        WHEEL_LEN = 0.3  # [m]
        WHEEL_WIDTH = 0.2  # [m]
        TREAD = 0.7  # [m]
        WB = 2.5  # [m]

        outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                            [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

        fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                            [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                            -WHEEL_WIDTH - TREAD]])

        rr_wheel = fr_wheel.copy()
        fl_wheel = fr_wheel.copy()
        rl_wheel = fr_wheel.copy()
        fl_wheel[1, :] *= -1
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]])

        fr_wheel = (fr_wheel.T @ Rot2).T
        fl_wheel = (fl_wheel.T @ Rot2).T
        fr_wheel[0, :] += WB
        fl_wheel[0, :] += WB

        fr_wheel = (fr_wheel.T @ Rot1).T
        fl_wheel = (fl_wheel.T @ Rot1).T
        outline = (outline.T @ Rot1).T
        rr_wheel = (rr_wheel.T @ Rot1).T
        rl_wheel = (rl_wheel.T @ Rot1).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        plt.plot(outline[0, :], outline[1, :], truckcolor)
        plt.plot(fr_wheel[0, :], fr_wheel[1, :], truckcolor)
        plt.plot(rr_wheel[0, :], rr_wheel[1, :], truckcolor)
        plt.plot(fl_wheel[0, :], fl_wheel[1, :], truckcolor)
        plt.plot(rl_wheel[0, :], rl_wheel[1, :], truckcolor)

    @staticmethod
    def plot_map(cones_by_type):
        """
        Plot cones based on their type (left or right).
        """
        cones_left = cones_by_type[ConeTypes.LEFT]
        cones_right = cones_by_type[ConeTypes.RIGHT]
        orange_big = cones_by_type[ConeTypes.ORANGE_BIG]
        orange_small = cones_by_type[ConeTypes.ORANGE_SMALL]
        cones_unknown = cones_by_type[ConeTypes.UNKNOWN]
        
        if len(cones_left) > 0:
            plt.plot(cones_left[:, 0], cones_left[:, 1], "ob", label="Left Cones",markersize=2)
        if len(cones_right) > 0:
            plt.plot(cones_right[:, 0], cones_right[:, 1], "oy", label="Right Cones",markersize=2)
        if len(cones_unknown) > 0:
            plt.plot(cones_unknown[:, 0], cones_unknown[:, 1], "og", label="Unkonwn Cones",markersize=2)
        if len(orange_big) > 0:
            plt.plot(orange_big[:, 0], orange_big[:, 1], color="orange", marker="o", label="orange big",markersize=2)
        if len(orange_small) > 0:
            plt.plot(orange_small[:, 0], orange_small[:, 1], color="orange", marker="o", label="orange",markersize=2)

    @staticmethod
    def plot_cones(cones_by_type, cones_lidar=[]):
        """
        Plot cones based on their type (left or right).
        """
        
        Visualizer.plot_map(cones_by_type)
        if len(cones_lidar) > 0:
            plt.plot(cones_lidar[:, 0], cones_lidar[:, 1], "og", label="Lidar Cones",markersize=2)

    @staticmethod
    def draw_frame(cx, cy, states, cones_by_type, target_ind, state, di, v_log, cones_lidar):
        """
        Draw a single frame of the animation.
        """
        plt.cla()
        plt.plot(cx, -cy, "r--", label="Planned Path", linewidth=2.5)
        plt.plot(states.x, states.y, "-c", label="Vehicle Path")
        Visualizer.plot_cones(cones_by_type, cones_lidar)
        plt.plot(cx[target_ind], -cy[target_ind], "xg", label="Target", markersize=10, linewidth=1)
        Visualizer.plot_car(state.x, -state.y, -state.yaw, steer=di)
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"Speed [km/h]: {state.v * 3.6:.2f}")
        plt.suptitle(f"target Speed [km/h]: {v_log * 3.6:.2f}")
        plt.pause(0.001)

    @staticmethod
    def show(cx, cy, states):
        """
        Show the final animation plot.
        """
        plt.figure()
        plt.title("Path Tracking")
        plt.plot(cx, -cy, "r--", label="Planned Path")
        plt.plot(states.x, states.y, "-b", label="Vehicle Path")
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    # Class variables to store the history of CTE and heading error
    cte_history = []
    theta_e_history = []

    @staticmethod
    def cross_track_error(e_ct, path, cx, cy, theta_e):
        """
        Store the cross track error (and optional heading error).
        
        Args:
            e_ct (float): The cross track error at the current timestep.
            path (numpy.ndarray): Path coordinates [[index, x, y], ...].
            cx (numpy.ndarray): X-coordinates of the path points.
            cy (numpy.ndarray): Y-coordinates of the path points.
            theta_e (float): Heading error at the current timestep.
        """
        Visualizer.cte_history.append(e_ct)
        Visualizer.theta_e_history.append(theta_e)

    @staticmethod
    def plot_cte(dt=0.05):
        """
        Plot the stored cross track errors over time.
        
        Args:
            dt (float): Time step between control updates. Adjust as needed.
        """
        if len(Visualizer.cte_history) == 0:
            print("No CTE data to plot.")
            return

        time = np.arange(0, len(Visualizer.cte_history) * dt, dt)
        
        plt.figure(figsize=(10,4))
        plt.plot(time, Visualizer.cte_history, label='Cross Track Error')
        plt.xlabel('Time [s]')
        plt.ylabel('CTE [m]')
        plt.title('Cross Track Error Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_path_deviation(cx, cy, states,paths):
        plt.figure()
        plt.plot(cx, -cy, label="Planned Path", linestyle="--", color="r")
        plt.plot(states.x, states.y, label="Actual Path", linestyle="-", color="b")
        plt.plot (paths[1:],-paths[2:],label ='planned path',linestyle="-", color="g")
        plt.title("Path Deviation")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_speed_profile(states, dt=conf.dt):
        # Match lengths of states.v and states.v_log
        min_length = min(len(states.v), len(states.v_log))
        time = np.arange(0, min_length * dt, dt)
        Target_speed_time = np.full_like(time, conf.TARGET_SPEED)
        plt.figure()
        plt.plot(time, states.v[:min_length], label="Actual Speed [m/s]", color='blue')
        plt.plot(time, states.v_log[:min_length], label="Target Speed (v_log) [m/s]", linestyle="--", color='red')
        plt.plot(time, Target_speed_time, label="Target Speed (Target Speed) [m/s]", linestyle="dashdot", color='Black',markersize = 5)
        plt.title("Speed Profile")
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_control_inputs(states, dt=conf.dt):
        # Match the lengths of all arrays
        min_length = min(len(states.t), len(states.steering), len(states.acceleration))
        time = np.arange(0, min_length * dt, dt)

        plt.figure()
        plt.plot(time, states.steering[:min_length], label="Steering Angle [rad]", color='green')
        plt.plot(time, states.acceleration[:min_length], label="Acceleration [m/s²]", color='orange')
        plt.title("Control Inputs Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Input")
        plt.legend()
        plt.grid()
        plt.show()