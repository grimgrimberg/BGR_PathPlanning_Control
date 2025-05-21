from __future__ import annotations

import itertools, math, logging, os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from core.data.plot_data import PlotData
from fsd_path_planning import ConeTypes
from vehicle_config import Vehicle_config as conf

log = logging.getLogger("PlotManager")

_FIG_COUNTER = itertools.count(1)        # fig_001.png, fig_002.png, …
_FIGS: List[Figure] = []
_LIVE = True #doesnt do shit as of right now.

def _auto_fig() -> Figure:
    """Create a figure, keep it for later saving, and return it."""
    fig = plt.figure()
    _FIGS.append(fig)
    return fig


def _maybe_show():
    if _LIVE:
        plt.pause(0.001)

class PlotManager:
    def __init__(self, live: bool):
        global _LIVE
        _LIVE = live
        # if live:
        #     plt.ion()
        # else:
        #     matplotlib.use("Agg")  # headless / non-interactive
        self.data = PlotData()  # single data object

        # # histories
        # self.cte_history: List[float] = []
        # self.theta_e_history: List[float] = []


        # ---------- life-cycle hooks ------------------------------------- #
    def update(self, _data: Dict[str, Any]):
        self.data = _data.get("plot_data", self.data)
        # if _LIVE:
        PlotManager.draw_frame(
            self.data.cx,
            self.data.cy,
            self.data.states,
            self.data.cones_map,
            self.data.target_ind,
            self.data.state,
            self.data.steering,
            self.data.v_log,
            self.data.cones_lidar,
            self.data.intermediate,
        )
    # ---------------------------------------------------------------- #
    @staticmethod
    def save_all(output_dir: str | Path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for idx, fig in enumerate(_FIGS, 1):
            fname = Path(output_dir) / f"fig_{idx:03d}.png"
            fig.savefig(fname, dpi=150)
            log.info("saved %s", fname)
        log.info("exported %d figures to %s", len(_FIGS), output_dir)


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
        PlotManager.plot_map(cones_by_type)
        if len(cones_lidar) > 0 and isinstance(cones_lidar, np.ndarray):
            plt.plot(cones_lidar[:, 0], cones_lidar[:, 1], "og", label="Lidar Cones",markersize=2,)

    # @staticmethod
    # def plot_route(path):
    #     cx, cy = path[:, 1], path[:, 2]
    #     plt.figure()
    #     plt.plot(cx, -cy, "r--", label="Planned Path", linewidth=2.5)
    #     plt.show()

    @staticmethod
    def draw_frame(cx, cy, states, cones_by_type, target_ind, state, di, v_log, cones_lidar, intermediate_results={}):
        """
        Draw a single frame of the animation.
        """
        plt.cla()
        plt.plot(cx, -cy, "r--", label="Planned Path", linewidth=2.5)
        plt.plot(states.x, states.y, "-c", label="Vehicle Path")
        PlotManager.plot_cones(cones_by_type, cones_lidar)
        plt.plot(cx[target_ind], -cy[target_ind], "xg", label="Target", markersize=10, linewidth=1)
        PlotManager.plot_car(state.x, -state.y, -state.yaw, steer=di)
        plt.plot(*intermediate_results["left_cones_with_virtual"].T, "o-", c="blue")
        plt.plot(*intermediate_results["right_cones_with_virtual"].T, "o-", c="yellow")
        plt.plot(*intermediate_results["left_cones_with_virtual"].T, "o-", c="blue")
        plt.plot(*intermediate_results["right_cones_with_virtual"].T, "o-", c="yellow")
        plt.plot(*intermediate_results["all_cones"].T, "o", c="k")
        for left, right_idx in zip(intermediate_results["left_cones_with_virtual"], intermediate_results["left_to_right_match"]):
            plt.plot(
                [left[0], intermediate_results["right_cones_with_virtual"][right_idx][0]],
                [left[1], intermediate_results["right_cones_with_virtual"][right_idx][1]],
                "-",
                c="#7CB9E8"
            )
        for right, left_idx in zip(intermediate_results["right_cones_with_virtual"], intermediate_results["right_to_left_match"]):
            plt.plot(
                [right[0], intermediate_results["left_cones_with_virtual"][left_idx][0]],
                [right[1], intermediate_results["left_cones_with_virtual"][left_idx][1]],
                "-",
                c="gold",
                alpha=0.5,
            )

        plt.axis("equal")
        plt.grid(True)
        plt.title(f"Speed [km/h]: {state.v * 3.6:.2f}")
        plt.suptitle(f"target Speed [km/h]: {v_log * 3.6:.2f}")
        plt.pause(0.00001)

    @staticmethod
    def show(cx, cy, states):
        """
        Show the final animation plot.
        """
        fig = _auto_fig()
        plt.figure(fig.number)
        plt.title("Path Tracking")
        plt.plot(cx, -cy, "r--", label="Planned Path")
        plt.plot(states.x, states.y, "-b", label="Vehicle Path")
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.grid(True)
        _maybe_show()

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
        PlotManager.cte_history.append(e_ct)
        PlotManager.theta_e_history.append(theta_e)

    @staticmethod
    def plot_cte(dt=0.05):
        """
        Plot the stored cross track errors over time.
        """
        if len(PlotManager.cte_history) == 0:
            print("No CTE data to plot.")
            return

        time = np.arange(0, len(PlotManager.cte_history) * dt, dt)

        fig = _auto_fig()
        plt.figure(fig.number)
        plt.plot(time, PlotManager.cte_history, label='Cross Track Error')
        plt.xlabel('Time [s]')
        plt.ylabel('CTE [m]')
        plt.title('Cross Track Error Over Time')
        plt.grid(True)
        plt.legend()
        _maybe_show()

    @staticmethod
    def plot_path_deviation(cx, cy, states, full_path):
        x, y = zip(*full_path)
        x = np.array(x)
        y = np.array(y)

        fig = _auto_fig()
        plt.figure(fig.number)
        plt.plot(cx, -cy, label="Planned Path", linestyle="--", color="r")
        plt.plot(states.x, states.y, label="Actual Path", linestyle="-", color="b")
        plt.plot(x, -y, label='planned path', linestyle="-", color="g", markersize=1)
        plt.title("Path Deviation")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid()
        _maybe_show()

    @staticmethod
    def plot_path_deviation1(cx, cy, states, X,Y,cones_by_type,cones_lidar):
        fig = _auto_fig()

        # x , y = zip(*full_path)
        # x , y = full_path[0],full_path[1]
        # x=full_path[:,1]
        # y=full_path[:,2]
        # Y=float(Y)
        # Y=Y*-1
        # sorted(x)
        # print("this is cx ",cx)
        # print("this is x ",x)
        # list(set(X))
        # list(set(Y))
        # X = np.unique(X, axis=0)
        # Y = np.unique(Y, axis=0)

# If X (and similarly Y) is a list of 1D arrays, flatten it:
        X_flat = np.concatenate(X)
        Y_flat = np.concatenate(Y)
        actual_x = np.array(states.x)
        actual_y = np.array(states.y)
        # Determine the number of points to compare
        n_points = min(len(actual_x), len(X_flat))

        # Slice both planned and actual arrays to the same length
        planned_x_trunc = X_flat[:n_points]
        planned_y_trunc = Y_flat[:n_points]
        actual_x_trunc  = actual_x[:n_points]
        actual_y_trunc  = actual_y[:n_points]

        # Calculate the MSE
        mse = np.mean((actual_x_trunc - planned_x_trunc)**2 + (actual_y_trunc - planned_y_trunc)**2)
        mse = mse/100
        rmse = np.sqrt(mse)

        print("MSE:", mse)


        # plt.figure()
        # plt.plot(cx, -cy, label="Planned Path", linestyle="--", color="r")
        plt.plot(states.x, states.y, label="Actual Path", linestyle="-", color="b")
        PlotManager.plot_cones(cones_by_type, cones_lidar)
        # plt.plot (X,Y,label ='planned path combined',linestyle="-", color="g", markersize=1)
        plt.plot(X_flat, Y_flat, label='planned path combined', linestyle="-", color="g", markersize=1)
        plt.title(f"Path Deviation (MSE: {mse:.2f})")
        # plt.text(0.05, 0.95, f"MSE: {mse:.2f}", transform=plt.gca().transAxes,
        #  fontsize=12, verticalalignment='top')
        plt.text(0.65, 0.65, f"RMSE: {rmse:.2f}", transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')
        plt.legend("lower center")


        # plt.title("Path Deviation")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid()
        # plt.show()
        _maybe_show()

    @staticmethod
    def plot_speed_profile(states, dt=conf.dt):
        min_length = min(len(states.v), len(states.v_log))
        time = np.linspace(0, (min_length - 1) * dt, min_length)
        Target_speed_time = np.full_like(time, conf.TARGET_SPEED)

        fig = _auto_fig()
        plt.figure(fig.number)
        plt.plot(time, states.v[:min_length], label="Actual Speed [m/s]", color='blue')
        plt.plot(time, states.v_log[:min_length], label="Target Speed (v_log) [m/s]", linestyle="--", color='red')
        plt.plot(time, Target_speed_time, label="Target Speed (Target Speed) [m/s]", linestyle="dashdot", color='Black',markersize = 5)
        plt.title("Speed Profile")
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        plt.legend()
        plt.grid()
        _maybe_show()

    @staticmethod
    def plot_control_inputs(states, dt=conf.dt):
        min_length = min(len(states.t), len(states.steering), len(states.acceleration))
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        fig = _auto_fig()
        plt.figure(fig.number)
        plt.plot(time, states.steering[:min_length], label="Steering Angle [rad]", color='green')
        plt.plot(time, states.acceleration[:min_length], label="Acceleration [m/s²]", color='orange')
        plt.title("Control Inputs Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Input")
        plt.legend()
        plt.grid()
        _maybe_show()


    @staticmethod
    def plot_point_cloud(points, title="3D Lidar Point Cloud"):
        """
        Visualize a 3D point cloud.
        """
        fig = _auto_fig()
        ax = fig.add_subplot(211, projection='3d')

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        ax.scatter(xs, ys, zs, c='b', marker='.', s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        _maybe_show()

    @staticmethod
    def plot_acceleration(states, dt=conf.dt):
        print("Plotting accelerations...")

        min_length = min(len(states.a_linear), len(states.a_angular))
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        fig = _auto_fig()
        plt.figure(fig.number)
        plt.plot(time, states.a_linear[:min_length], label="Linear Acceleration [m/s²]", color='green')
        plt.plot(time, states.a_angular[:min_length], label="Angular Acceleration [rad/s²]", color='orange')
        plt.title("Accelerations Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid()
        _maybe_show()

    @staticmethod
    def plot_gg(states, dt=conf.dt):
        print("Plotting GG Diagram clearly with stored numeric data...")

        longitudinal_accel = np.array(states.a_longitudinal)/9.81
        lateral_accel = np.array(states.a_lateral)/9.81

        fig = _auto_fig()
        plt.figure(fig.number, figsize=(8, 8))
        plt.scatter(lateral_accel, longitudinal_accel, s=5, c='blue', alpha=0.5)

        max_accel = max(np.max(np.abs(longitudinal_accel)), np.max(np.abs(lateral_accel))) + 1
        plt.xlim(-max_accel, max_accel)
        plt.ylim(-max_accel, max_accel)

        plt.xlabel("Lateral Acceleration [m/s²]")
        plt.ylabel("Longitudinal Acceleration [m/s²]")
        plt.title("GG Diagram (Friction Circle)")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        _maybe_show()

    @staticmethod
    def plot_all_accelerations(states, dt=conf.dt):
        print("Plotting longitudinal, lateral, and angular accelerations explicitly...")

        min_length = min(len(states.a_longitudinal), len(states.a_lateral), len(states.v_angular))
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        angular_velocity = np.array(states.v_angular)
        angular_acceleration = np.gradient(angular_velocity, dt)

        fig = _auto_fig()
        plt.figure(fig.number, figsize=(12, 6))
        plt.plot(time, states.a_longitudinal[:min_length], label="Longitudinal Acceleration [m/s²]", color='green')
        plt.plot(time, states.a_lateral[:min_length], label="Lateral Acceleration [m/s²]", color='blue')
        plt.plot(time, angular_acceleration[:min_length], label="Angular Acceleration [rad/s²]", color='orange')

        plt.title("Longitudinal, Lateral, and Angular Accelerations Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid(True)
        _maybe_show()

    @staticmethod
    def plot_point_cloud(points, title="3D Lidar Point Cloud"):
        """
        Visualize a 3D point cloud.
        """
        fig = _auto_fig()
        ax = fig.add_subplot(211, projection='3d')

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        ax.scatter(xs, ys, zs, c='b', marker='.', s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        _maybe_show()

    # @staticmethod
    # def plot_acceleration(a_linear,a_angular,dt=conf.dt):
    #     print("im here plotting the accel")
    #     min_length = min(len(a_angular), len(a_linear))
    #     time = np.arange(0, min_length * dt, dt)
    #     plt.figure()
    #     plt.plot(time, a_linear[:min_length], label="Linear Accelration [m/s²]", color='green')
    #     plt.plot(time, a_angular[:min_length], label="Angular Accelration [Rad/s²]", color='orange')
    #     plt.title("Accelerations Over Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Accelerations")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    #     return
    # @staticmethod
    # def plot_intermediate_results(out,lidar_cones_by_type,car_position, car_direction):
    #     # cones_left, cones_right, cones_unknown = lidar_cones_by_type.values()
    #     plt.scatter(cones_left[:, 0], cones_left[:, 1], c=blue_color, label="left")
    #     plt.scatter(cones_right[:, 0], cones_right[:, 1], c=yellow_color, label="right")
    #     plt.scatter(cones_unknown[:, 0], cones_unknown[:, 1], c="k", label="unknown")

    #     plt.legend()

    #     plt.plot(
    #         [car_position[0], car_position[0] + car_direction[0]],
    #         [car_position[1], car_position[1] + car_direction[1]],
    #         c="k",
    #     )
    #     plt.title("Computed path")
    #     plt.plot(*out[:, 1:3].T)

    #     plt.axis("equal")

    #     plt.show()

    #     plt.title("Curvature over distance")
    #     plt.plot(out[:, 0], out[:, 3])

    #     ##
    #     all_cones = np.row_stack([cones_left, cones_right, cones_unknown])


    #     plt.plot(*all_cones.T, "o", c="k")
    #     plt.plot(*sorted_left.T, "o-", c=blue_color)
    #     plt.plot(*sorted_right.T, "o-", c=yellow_color)
    #     plt.title("Sorted cones")
    #     plt.axis("equal")
    #     plt.show()


    #     plt.plot(*all_cones.T, "o", c="k")
    #     plt.plot(*left_cones_with_virtual.T, "o-", c=blue_color)
    #     plt.plot(*right_cones_with_virtual.T, "o-", c=yellow_color)
    #     plt.title("Left and right cones with virtual cones")
    #     plt.axis("equal")
    #     plt.show()

    #     plt.plot(*all_cones.T, "o", c="k")


    #     for left, right_idx in zip(left_cones_with_virtual, left_to_right_match):
    #         plt.plot(
    #             [left[0], right_cones_with_virtual[right_idx][0]],
    #             [left[1], right_cones_with_virtual[right_idx][1]],
    #             "-",
    #             c=blue_color,
    #         )


    #     for right, left_idx in zip(right_cones_with_virtual, right_to_left_match):
    #         plt.plot(
    #             [right[0], left_cones_with_virtual[left_idx][0]],
    #             [right[1], left_cones_with_virtual[left_idx][1]],
    #             "-",
    #             c=yellow_color,
    #             alpha=0.5,
    #         )

    #     plt.title("Left and right matches")
    #     plt.axis("equal")
    #     plt.show()

    #     ###
