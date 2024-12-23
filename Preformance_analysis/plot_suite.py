import matplotlib.pyplot as plt

def plot_path_tracking(data):
    plt.figure()
    plt.plot(data["x"], data["y"], label="Actual Path")
    plt.title("Path Tracking")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

def plot_lateral_error(data):
    plt.figure()
    plt.plot(data["timestamp"], data["lateral_error"])
    plt.title("Lateral Error Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral Error (m)")
    plt.grid()
    plt.show()

def plot_speed_profile(data):
    plt.figure()
    plt.plot(data["timestamp"], data["velocity"])
    plt.title("Speed Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid()
    plt.show()
