# BGR\_PathPlanning\_Control

BGR Autonomous Racing Team's GitHub repository for controllers, path planners, and simulation environments.

## Overview

A collection of path planning and control modules for Formula Student Autonomous (FSAE) vehicles, built on top of the `fsd-path-planning` library by FaSTTUBe. This repository provides additional scripts, simulation tools, and control algorithms to optimize your vehicle’s racing performance in missions like Trackdrive and Skidpad.

## Table of Contents

1. [Overview](#overview)
2. [Features & Missions](#features--missions)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Basic Usage](#basic-usage)
7. [Integration with fsd-path-planning](#integration-with-fsd-path-planning)
8. [Example Workflow](#example-workflow)
9. [Contributing](#contributing)
10. [License](#license)

## Features & Missions

- **Trackdrive**: Advanced logic for robust path generation—even with partial cone visibility.
- **Skidpad**: Improved handling of color-agnostic cones and integrated relocalization.
- **Caching & Performance**: Experiment with `fsd-path-planning` caching features for speed gains in real-time.
- **Flexible Controller Integration**: Utilize paths from `fsd-path-planning` in various tracking controllers to compare performance or switch strategies quickly.

## Repository Structure

```
BGR_PathPlanning_Control/
├── src/
│   ├── planners/       # Custom wrappers or enhancements for fsd-path-planning
│   ├── controllers/    # Control algorithms (e.g., Pure Pursuit, LQR, MPC)
│   ├── utils/          # Common vehicle models, coordinate transforms, etc.
│   └── main.py         # Example script demonstrating usage
├── scripts/            # Additional scripts for simulation and data processing
├── examples/           # Example notebooks and demos
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Prerequisites

- **Python**: 3.x (Recommended: 3.8+)
- **Environment Management**: `pip`, `virtualenv`, or `conda` (optional but recommended)
- **fsd-path-planning**: Ensure this library is installed (follow its official instructions).

## Installation

1. Clone this repository:
   ```bash
   git clone --branch yuval https://github.com/grimgrimberg/BGR_PathPlanning_Control.git
   cd BGR_PathPlanning_Control
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   pip install -r requirements.txt
   ```

## Basic Usage

### Prepare Your Cone/Vehicle Data

- Detect cone positions (optionally with colors).
- Acquire the vehicle’s pose in the SLAM map (e.g., x, y coordinates and heading).

### Use `fsd-path-planning`

```python
from fsd_path_planning import PathPlanner, MissionTypes

# Example usage for Trackdrive:
path_planner = PathPlanner(MissionTypes.trackdrive)

# Suppose you have global_cones, car_position, and car_direction from your pipeline:
path = path_planner.calculate_path_in_global_frame(
    global_cones,    # Array of shape (N, 2)
    car_position,    # e.g., np.array([x, y])
    car_direction    # e.g., float in radians, or a 2D direction vector
)
```

### Apply a Control Algorithm

```python
from src.controllers import pure_pursuit

steering_angle, acceleration = pure_pursuit.compute_control_commands(
    path,
    current_vehicle_state
)
# Send these commands to your simulator or real-time control stack.
```

### Run a Script or Demo

```bash
python scripts/run_bgr_demo.py --mission trackdrive
```

## Integration with fsd-path-planning

Key features of `fsd-path-planning`:

- **Color Modes**: Switch between color-dependent and color-agnostic modes.
- **Caching**: A new caching mechanism for sorting cones (use with caution if untested).
- **Relocalization Info**: For Skidpad, extract translation/rotation data for better alignment.

Our repository enhances these features by:

- Incorporating advanced parameters (e.g., `experimental_performance_improvements`).
- Demonstrating when to initialize new `PathPlanner` instances for stateful missions.
- Providing additional performance metrics to tune controller parameters.

### Important Notes

For Skidpad missions, avoid reinitializing the `PathPlanner` object on every iteration, as some internal state is maintained to ensure consistent path generation.

## Example Workflow

1. **Cone Detection**: Identify cone positions (and optionally colors) in real-time.
2. **Path Planning**: Use cone data and vehicle pose to calculate a path.
3. **Control**: Feed the path into a control algorithm (e.g., Pure Pursuit, LQR).
4. **Execution**: Send steering and throttle commands to the simulator or vehicle.
5. **Loop**: Update vehicle state from sensors or simulation and repeat from Step 2.

## Contributing

Contributions are welcome! To propose changes:

1. Fork the repository and create a feature branch.
2. Make your changes (add tests and update docs if needed).
3. Open a pull request describing your modifications.

Formula Student teams using this repository are encouraged to share their improvements and experiences.

## License

This repository is provided under the MIT License. See the `fsd-path-planning` license for additional details.

---

Happy racing, and best of luck on your Formula Student Driverless journey! If you have any questions, feel free to open an issue or contact the maintainers.

