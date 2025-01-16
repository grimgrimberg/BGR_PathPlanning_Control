# BGR_PathPlanning_Control
BGR Autonomous racing team  github that includes controllers, path planner, controllers and a sim enviorments

BGR_PathPlanning_Control
A collection of path planning and control modules for Formula Student Autonomous (FSAE) vehicles, built on top of fsd-path-planning by FaSTTUBe. This repository provides additional scripts, simulation tools, and control algorithms to streamline your vehicle’s racing performance in Trackdrive, Skidpad, and other missions.

Table of Contents
Overview
Features & Missions
Repository Structure
Prerequisites
Installation
Basic Usage
Integration with fsd-path-planning
Example Workflow
Contributing
License
Overview
This repository serves as an extension of the fsd-path-planning library maintained by FaSTTUBe. We provide:

Additional Scripts/Tools for racing lines, real-time path updates, and transitions between different missions (e.g., Trackdrive, Skidpad).
Control Algorithms (e.g., Pure Pursuit, LQR, MPC) with vehicle-specific parameters to track the planned path.
Simulation/Visualization utilities to test your path-planning & control loops.
Performance Enhancements or custom features that adapt to your Formula Student team’s pipeline.
Features & Missions
Trackdrive: Incorporates the advanced logic from fsd-path-planning for robust path generation—even with partial cone visibility.
Skidpad: Uses the improved Skidpad approach from fsd-path-planning, handling color-agnostic cones and integrated relocalization.
Caching & Performance: Allows you to experiment with fsd-path-planning’s caching features for potential speed gains in real-time.
Flexible Controller Integration: The path from fsd-path-planning can feed into various tracking controllers, letting you compare performance or quickly switch control strategies.
Repository Structure
bash
Copy code
BGR_PathPlanning_Control/
├── src/
│   ├── planners/       # Custom wrappers or enhancements on fsd-path-planning
│   ├── controllers/    # Control algorithms (e.g., Pure Pursuit, LQR, MPC)
│   ├── utils/          # Common vehicle models, coordinate transforms, etc.
│   └── main.py         # Example main script demonstrating usage
├── scripts/            # Additional scripts for simulation, data processing
├── examples/           # Example notebooks/demos showing how to integrate
├── tests/              # Unit tests and integration tests
├── requirements.txt    # Python dependencies
└── README.md           # This file
Adjust the folder names and file references to match your actual structure.

Prerequisites
Python 3.x (Recommended: 3.8+)
Pip / virtualenv / conda (optional, but recommended for environment management)
fsd-path-planning
Your environment must have a working installation of fsd-path-planning.
If you haven’t installed it yet, please follow their instructions.
Installation
Clone this Repository

bash
Copy code
git clone --branch yuval https://github.com/grimgrimberg/BGR_PathPlanning_Control.git
cd BGR_PathPlanning_Control
Install Dependencies

Ensure you have fsd-path-planning installed.
Then install any local Python dependencies:
bash
Copy code
pip install -r requirements.txt
(Optional) Create a Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
Basic Usage
Below is a high-level example of how you might run path planning and then feed the resulting path into a simple controller. Modify file names, function calls, or CLI commands based on your code structure.

Prepare Your Cone / Vehicle Data

Obtain cone positions (potentially with colors if available).
Acquire the vehicle’s pose in the SLAM map (x,y coordinates and heading).
Use fsd-path-planning

python
Copy code
from fsd_path_planning import PathPlanner, MissionTypes

# Example usage for Trackdrive:
path_planner = PathPlanner(MissionTypes.trackdrive)

# Suppose you have global_cones, car_position, and car_direction from your pipeline
path = path_planner.calculate_path_in_global_frame(
    global_cones,    # 5 arrays of shape (N,2)
    car_position,    # e.g. np.array([x, y])
    car_direction    # e.g. float in radians, or a 2D direction vector
)
Apply a Control Algorithm

python
Copy code
from src.controllers import pure_pursuit

steering_angle, acceleration = pure_pursuit.compute_control_commands(
    path, 
    current_vehicle_state
)
# Then pass these commands to your simulator or real-time control stack
Run a Script or Demo

bash
Copy code
python scripts/run_bgr_demo.py --mission trackdrive
This might, for instance, load the cone data, call the planner, run a controller, and visualize the results.

Integration with fsd-path-planning
FaSTTUBe’s path planning algorithm is continuously evolving with features like:

Color vs. No-Color: You can switch between color-dependent or color-agnostic modes.
Caching & Performance: A new caching mechanism can speed up sorting (use at your own risk if not fully tested).
Relocalization Info: For Skidpad, you can extract translation/rotation data from the path planner for better map alignment.
In BGR_PathPlanning_Control, we leverage these functionalities by:

Incorporating advanced parameters (e.g., experimental_performance_improvements) inside our scripts.
Demonstrating when to create a new PathPlanner instance (particularly important for Skidpad’s stateful path calculation).
Offering additional performance metrics or logs to help tune your controller parameters.
Important: For Skidpad, avoid re-initializing the PathPlanner object on every iteration—some internal state is maintained to ensure consistent path generation.

Example Workflow
Cone Detection
Your perception pipeline identifies cone positions (and possibly colors) in real-time.
Path Planning
Pass cone data & vehicle pose to fsd-path-planning.
Receive a B-spline (or set of path points with curvature) as output.
Control
Feed the path into a control algorithm (Pure Pursuit, LQR, etc.) from src/controllers/.
Execution
Send steering and throttle commands to your simulator or physical vehicle.
Receive updated vehicle state from sensors or simulator; loop back to step 2.
Contributing
Contributions are welcome! If you want to suggest improvements, fix bugs, or add new mission logic:

Fork this repository and create a branch for your changes.
Make your changes (including tests or updated docs when needed).
Open a pull request describing your modifications and rationale.
We encourage any formula student team using this repository to share their experiences or improvements.

License
Unless otherwise noted, this repository is provided under the MIT License. Please check the fsd-path-planning license for additional details on FaSTTUBe’s package.

Happy Racing and best of luck on your Formula Student Driverless journey! If you have questions, feel free to open an issue or reach out to the maintainers.
