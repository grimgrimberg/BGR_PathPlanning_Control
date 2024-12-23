import csv

class SimulationLogger:
    def __init__(self, filename="simulation_log.csv"):
        self.filename = filename
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "x", "y", "velocity", "steering_angle", 
                             "throttle", "brake", "lateral_error", "heading_error", 
                             "yaw_rate", "lateral_accel"])

    def log(self, timestamp, x, y, velocity, steering_angle, throttle, brake, 
            lateral_error, heading_error, yaw_rate, lateral_accel):
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, x, y, velocity, steering_angle, throttle, 
                             brake, lateral_error, heading_error, yaw_rate, lateral_accel])

    def close(self):
        pass  # Placeholder in case of additional cleanup logic
