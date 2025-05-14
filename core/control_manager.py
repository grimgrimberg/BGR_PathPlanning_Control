import time
import logging
from typing import List, Protocol, Any
from core.data.plot_data import PlotData
from core.visualization import PlotManager
log = logging.getLogger("ControlManager")


class ControlManager:
    def __init__(self, providers, subscribers, dt, enable_plots, output_dir):
        self.providers = providers
        self.subscribers = subscribers
        self.dt = dt
        self.enable_plots = enable_plots
        self.plot_data = PlotData()             # NEW: single data object
        self.plotter = PlotManager(live=enable_plots)
        self.output_dir = output_dir

    # ---------- public API ----------
    def run(self):
        log.info("Starting main loop dt=%.3f s (plots=%s)",
                 self.dt, self.enable_plots)
        for p in self.providers:   p.start()
        for s in self.subscribers: s.init(self.providers)

        t_prev = time.perf_counter()
        try:
            while True:
                now = time.perf_counter()
                if now - t_prev < self.dt:
                    time.sleep(0.001)
                    continue
                dt = now - t_prev
                t_prev = now

                data = {}
                for p in self.providers:
                    data.update(p.read())

                for s in self.subscribers:
                    s.update(data, dt)
                    
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
            log.info("Stopping …")
        # finally:
            # for s in self.subscribers: s.finish()
            # for p in self.providers:   p.stop()

    # ---------- optional ----------
    def run_calibration(self):
        log.info("Calibration routine (placeholder)")
        # your calibration steps here
