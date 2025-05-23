#!/usr/bin/env python3
"""
Run either a simulation or calibration session with a pluggable provider/
nodes pipeline.  Use --help for options.
"""
import sys
import argparse
import logging
from core.manager import Manager
from providers.sim.car_state import SimCarStateProvider
from providers.sim.cones     import SimConeProvider
from providers.sim.map_data  import SimMapProvider
from providers.lidar.cones   import LidarConeProvider
# Swap these two for ROS2 later:
# from providers.ros.car_state import ROSCarStateProvider
# from providers.ros.cones     import ROSConeProvider
# from providers.ros.map_data  import ROSMapProvider
from nodes.planner     import Planner
from nodes.controller  import Controller
from core.logger import init_logger



def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modular path-planning & control framework")
    p.add_argument("--mode",    choices=["simulation", "calibration"],
                   default="simulation")
    p.add_argument("--plot",    action="store_true",
                   help="Enable live plots / post-run analysis")
    p.add_argument("--dt", type=float, default=0.05,
                   help="Main-loop period [s]")
    p.add_argument("--output_dir", type=str, default="output",
                   help="Output directory for plots and data")
    return p.parse_args()

def build_manager(args):
    # Providers (simulation flavour)
    providers = [
        SimCarStateProvider(),      # vehicle pose & twist
        # LidarConeProvider(),          # LiDAR-derived track cones
        SimConeProvider(),          # simulated cones
        SimMapProvider()
    ]

    nodes = [
        Planner(),               # produces path based on cones & map
        Controller()             # produces throttle / steer commands
    ]

    return Manager(providers, nodes, dt=args.dt, enable_plots=args.plot, output_dir=args.output_dir)

def main():
    args = cli()
    # logging.basicConfig(level=logging.INFO,
    #                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_logger()
    mgr = build_manager(args)

    if args.mode == "calibration":
        mgr.run_calibration()
    else:
        mgr.run()

if __name__ == "__main__":
    main()
