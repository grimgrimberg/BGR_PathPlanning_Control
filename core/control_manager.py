import time
import logging
from typing import List, Protocol, Any
from core.data.plot_data import PlotData
from core.visualization import PlotManager

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
        self.output_dir = output_dir
        
        log.debug(f"Configuration - dt: {dt}, plots enabled: {enable_plots}, "
                 f"output directory: {output_dir}")
        log.debug(f"Initialized with {len(providers)} providers and {len(nodes)} nodes")

    # ---------- public API ----------
    def run(self):
        log.info(f"Starting main control loop - dt={self.dt:.3f}s, plots={self.enable_plots}")
        
        try:
            # Initialize providers
            for p in self.providers:
                try:
                    p.start()
                    log.debug(f"Provider {p.__class__.__name__} started successfully")
                except Exception as e:
                    log.error(f"Failed to start provider {p.__class__.__name__}: {str(e)}", 
                            exc_info=True)
                    raise
            
            # Initialize nodes
            for n in self.nodes:
                try:
                    n.init(self.providers)
                    log.debug(f"Node {n.__class__.__name__} initialized successfully")
                except Exception as e:
                    log.error(f"Failed to initialize node {n.__class__.__name__}: {str(e)}", 
                            exc_info=True)
                    raise

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

                # Update nodes
                try:
                    for n in self.nodes:
                        n.update(data, dt)
                        log.debug(f"Updated node {n.__class__.__name__}")
                except Exception as e:
                    log.error(f"Error updating nodes: {str(e)}", exc_info=True)
                    raise
                    
                # Update plot data
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

                if self.enable_plots or True:
                    self.plotter.update({"plot_data": self.plot_data})
                
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt, stopping control loop")
        except Exception as e:
            log.error(f"Unexpected error in control loop: {str(e)}", exc_info=True)
            raise
        finally:
            self._cleanup()

    # ---------- optional ----------
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
