from performance_analysis.data_loader import load_simulation_data
from performance_analysis.plot_suite import (
    plot_path_tracking, 
    plot_lateral_error, 
    plot_speed_profile
)
from performance_analysis.metrics_calculator import (
    calculate_lateral_error, 
    calculate_heading_error, 
    calculate_lap_time
)

def evaluate_simulation(log_file):
    """Evaluate simulation performance from logged data."""
    data = load_simulation_data(log_file)

    # Calculate and print metrics
    mean_lateral_error, std_lateral_error = calculate_lateral_error(data)
    mean_heading_error, std_heading_error = calculate_heading_error(data)
    lap_time = calculate_lap_time(data)

    print(f"Lap Time: {lap_time:.2f} seconds")
    print(f"Lateral Error: Mean={mean_lateral_error:.2f}, Std={std_lateral_error:.2f}")
    print(f"Heading Error: Mean={mean_heading_error:.2f}, Std={std_heading_error:.2f}")

    # Generate plots
    plot_path_tracking(data)
    plot_lateral_error(data)
    plot_speed_profile(data)
