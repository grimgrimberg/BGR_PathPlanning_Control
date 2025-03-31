import math
import numpy as np
import matplotlib.pyplot as plt
from fsd_path_planning import ConeTypes
from vehicle_config import Vehicle_config as conf
# from animation_loop import v_log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

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
            plt.plot(cones_lidar[:, 0], cones_lidar[:, 1], "og", label="Lidar Cones",markersize=2,)

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
        plt.pause(0.00001)

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
    def plot_path_deviation(cx, cy, states, full_path):
        x , y = zip(*full_path)
        x=np.array(x)
        y=np.array(y)
        # x , y = full_path[0],full_path[1]
        # x=full_path[:,1]
        # y=full_path[:,2]
        
        # sorted(x)
        # print("this is cx ",cx)
        # print("this is x ",x)
        plt.figure()
        plt.plot(cx, -cy, label="Planned Path", linestyle="--", color="r")
        plt.plot(states.x, states.y, label="Actual Path", linestyle="-", color="b")
        plt.plot (x,-y,label ='planned path',linestyle="-", color="g", markersize=1)
        plt.title("Path Deviation")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid()
        plt.show()
        
    @staticmethod
    def plot_path_deviation1(cx, cy, states, X,Y,cones_by_type,cones_lidar):
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
        rmse = np.sqrt(mse)/100
        mse = mse/100
        
        print("MSE:", mse)


        plt.figure()
        # plt.plot(cx, -cy, label="Planned Path", linestyle="--", color="r")
        plt.plot(states.x, states.y, label="Actual Path", linestyle="-", color="b")
        Visualizer.plot_cones(cones_by_type, cones_lidar)
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
        plt.show()

    @staticmethod
    def plot_speed_profile(states, dt=conf.dt):
        # Match lengths of states.v and states.v_log
        min_length = min(len(states.v), len(states.v_log))
        # time = np.arange(0, min_length * dt, dt)
        time = np.linspace(0, (min_length - 1) * dt, min_length)
        Target_speed_time = np.full_like(time, conf.TARGET_SPEED)
        print(len(states.v[:min_length]))
        print(len(states.v_log[:min_length]))
        print(len(Target_speed_time))
        
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
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        # time = np.arange(0, min_length * dt, dt)

        plt.figure()
        plt.plot(time, states.steering[:min_length], label="Steering Angle [rad]", color='green')
        plt.plot(time, states.acceleration[:min_length], label="Acceleration [m/s²]", color='orange')
        plt.title("Control Inputs Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Input")
        plt.legend()
        plt.grid()
        plt.show()
        

    @staticmethod
    def plot_point_cloud(points, title="3D Lidar Point Cloud"):
        """
        Visualize a 3D point cloud.

        Args:
            points (np.ndarray): A NumPy array of shape (N, 3) representing the point cloud.
            title (str): Title for the plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        
        # Assuming points is an array with columns [x, y, z]
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        
        # ax.scatter(ys, -xs, zs, s=1, c='b', marker='.')
        ax.scatter(xs, ys, zs, s=1, c='b', marker='.')#orig
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()
        plt.pause(0.001)
        
    
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
    @staticmethod
    def plot_acceleration(states, dt=conf.dt):
        print("Plotting accelerations...")

        min_length = min(len(states.a_linear), len(states.a_angular))
        # time = np.arange(0, min_length * dt, dt)
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        plt.figure()
        plt.plot(time, states.a_linear[:min_length], label="Linear Acceleration [m/s²]", color='green')
        plt.plot(time, states.a_angular[:min_length], label="Angular Acceleration [rad/s²]", color='orange')
        plt.title("Accelerations Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid()
        plt.show()
        
    @staticmethod
    def plot_gg(states, dt=conf.dt):
        print("Plotting GG Diagram clearly with stored numeric data...")

        longitudinal_accel = np.array(states.a_longitudinal)/9.81
        lateral_accel = np.array(states.a_lateral)/9.81

        plt.figure(figsize=(8, 8))
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
        plt.show()

    @staticmethod
    def plot_all_accelerations(states, dt=conf.dt):
        print("Plotting longitudinal, lateral, and angular accelerations explicitly...")

        min_length = min(len(states.a_longitudinal), len(states.a_lateral), len(states.v_angular))
        time = np.linspace(0, (min_length - 1) * dt, min_length)

        # Explicitly compute angular acceleration from angular velocity (yaw-rate)
        angular_velocity = np.array(states.v_angular)
        angular_acceleration = np.gradient(angular_velocity, dt)

        plt.figure(figsize=(12, 6))
        plt.plot(time, states.a_longitudinal[:min_length], label="Longitudinal Acceleration [m/s²]", color='green')
        plt.plot(time, states.a_lateral[:min_length], label="Lateral Acceleration [m/s²]", color='blue')
        plt.plot(time, angular_acceleration[:min_length], label="Angular Acceleration [rad/s²]", color='orange')

        plt.title("Longitudinal, Lateral, and Angular Accelerations Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    # @staticmethod
    # def plot_point_cloud(points, title="3D Lidar Point Cloud"):
    #     """
    #     Visualize a 3D point cloud interactively.

    #     Args:
    #         points (np.ndarray): A NumPy array of shape (N, 3).
    #         title (str): Plot title.
    #     """
    #     plt.figure("3D Lidar Point Cloud", figsize=(8, 6))
    #     plt.clf()  # Clear figure explicitly
    #     ax = plt.axes(projection='3d')

    #     # Explicit coordinate extraction
    #     xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

    #     # Explicit scatter plotting
    #     ax.scatter(xs, ys, zs, s=2, c='b', marker='.')

    #     ax.set_xlabel('X [m]')
    #     ax.set_ylabel('Y [m]')
    #     ax.set_zlabel('Z [m]')
    #     ax.set_title(title)

    #     plt.draw()
    #     plt.pause(0.01)  # Explicit GUI update
