import time
import logging
from typing import Dict, List, Protocol, Any
from core.data.plot_data import PlotData
from core.visualization import PlotManager
from pathlib import Path
from core.logger import log_timing

log = logging.getLogger("ControlManager")

class ControlManager:
    def __init__(self, providers, nodes, dt, enable_plots, output_dir):
        log.info("Initializing Control Manager")
        self.providers = providers
        self.nodes = nodes
        self.dt = dt
        self.enable_plots = enable_plots
        self.plot_data = PlotData()
        self.plotter = PlotManager(live=enable_plots)
        self.output_dir = Path(output_dir)
        
        log.debug(f"Configuration - dt: {dt}, plots enabled: {enable_plots}, "
                 f"output directory: {output_dir}")
        log.debug(f"Initialized with {len(providers)} providers and {len(nodes)} nodes")

    def run(self):
        log.info(f"Starting main control loop - dt={self.dt:.3f}s, plots={self.enable_plots}")
        
        try:
            t_prev = time.perf_counter()
            iteration = 0
            
            while True:
                iteration += 1
                now = time.perf_counter()
                
                # Timing control
                if now - t_prev < self.dt:
                    time.sleep(0.001)
                    continue
                
                dt = now - t_prev
                t_prev = now
                
                if dt > self.dt * 1.5:  # Warn if loop takes too long
                    log.warning(f"Loop iteration {iteration} took {dt:.3f}s "
                              f"(target: {self.dt:.3f}s)")
                # Read from providers
                t_read_providers = time.perf_counter()
                try:
                    data = {}
                    for p in self.providers:
                        provider_data = p.read()
                        log.debug(f"Read from {p.__class__.__name__}: "
                                f"{len(provider_data) if provider_data else 0} items")
                        data.update(provider_data)
                except Exception as e:
                    log.error(f"Error reading from providers: {str(e)}", exc_info=True)
                    raise
                
                log_timing("read", time.perf_counter() - t_read_providers)
                # Update nodes
                t_update_nodes = time.perf_counter()
                try:
                    for n in self.nodes:
                        n.update(data, dt)
                        log.debug(f"Updated node {n.__class__.__name__}")
                except Exception as e:
                    log.error(f"Error updating nodes: {str(e)}", exc_info=True)
                    raise
                log_timing("update_nodes", time.perf_counter() - t_update_nodes)
                data["current_time"] = now 


                if self.enable_plots:
                    self._update_plot_data(data)
                    try:
                        self.plotter.update({"plot_data": self.plot_data})
                        log.debug("Plot data updated successfully")
                    except Exception as e:
                        log.error(f"Error updating plots: {str(e)}", exc_info=True)
                
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt, stopping control loop")
            self._generate_final_plots()
        except Exception as e:
            log.error(f"Unexpected error in control loop: {str(e)}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def run_calibration(self):
        log.info("Starting calibration routine")
        # Implement calibration steps here
        pass

    def _cleanup(self):
        """Clean up resources and shut down components."""
        log.info("Cleaning up resources")
        try:
            for n in self.nodes:
                try:
                    if hasattr(n, 'finish'):
                        n.finish()
                    log.debug(f"Node {n.__class__.__name__} finished successfully")
                except Exception as e:
                    log.error(f"Error finishing node {n.__class__.__name__}: {str(e)}", 
                            exc_info=True)
            
            for p in self.providers:
                try:
                    if hasattr(p, 'stop'):
                        p.stop()
                    log.debug(f"Provider {p.__class__.__name__} stopped successfully")
                except Exception as e:
                    log.error(f"Error stopping provider {p.__class__.__name__}: {str(e)}", 
                            exc_info=True)
        except Exception as e:
            log.error(f"Error during cleanup: {str(e)}", exc_info=True)
    
    def _generate_final_plots(self):
        """Generate and save all plots at the end of the run."""
        log.info("Generating final plots...")
        
        try:
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate all available plots
            if self.plot_data.states is not None and self.plot_data.cx is not None and self.plot_data.cy is not None:
                # Path tracking plot
                self.plotter.show(self.plot_data.cx, self.plot_data.cy, self.plot_data.states)
                
                # Speed profile
                self.plotter.plot_speed_profile(self.plot_data.states)
                
                # Control inputs
                self.plotter.plot_control_inputs(self.plot_data.states)
                
                # Path deviation
                if self.plot_data.full_path:
                    self.plotter.plot_path_deviation(
                        self.plot_data.cx,
                        self.plot_data.cy,
                        self.plot_data.states,
                        self.plot_data.full_path
                    )
                # TODO: add to carState all atributes needed for the plots
                # # Acceleration plots
                # self.plotter.plot_all_accelerations(self.plot_data.states)
                # self.plotter.plot_gg(self.plot_data.states)
                
                # Cross track error
                if hasattr(self.plotter, 'cte_history') and len(self.plotter.cte_history) > 0:
                    self.plotter.plot_cte()
            
            # Save all generated plots
            self.plotter.save_all(plots_dir)
            log.info(f"All plots saved to {plots_dir}")
            
        except Exception as e:
            log.error(f"Error generating final plots: {str(e)}", exc_info=True)

    def _update_plot_data(self, data: Dict[str, Any]):
        """Update plot data from node outputs."""
        self.plot_data.state = data.get("car_state")
        self.plot_data.states = data.get("states")
        self.plot_data.path = data.get("path")
        self.plot_data.cones_map = data.get("cones_map")
        self.plot_data.cones_lidar = data.get("cones_lidar")
        self.plot_data.cx = data.get("cx")
        self.plot_data.cy = data.get("cy")
        self.plot_data.acceleration = data.get("acceleration")
        self.plot_data.steering = data.get("steering")
        self.plot_data.target_ind = data.get("target_ind")
        self.plot_data.v_log = data.get("v_log")
        self.plot_data.full_path = data.get("full_path", [])
