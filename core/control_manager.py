import time
import logging
from typing import List, Protocol, Any

log = logging.getLogger("ControlManager")

class Provider(Protocol):
    """Anything that yields data each cycle."""
    def start(self): ...
    def stop(self): ...
    def read(self) -> dict: ...

class Subscriber(Protocol):
    """Anything that updates internal state each cycle using provider data."""
    def init(self, providers: List[Provider]): ...
    def update(self, data: dict, dt: float): ...
    def finish(self): ...

class ControlManager:
    def __init__(self,
                 providers:  List[Provider],
                 subscribers:List[Subscriber],
                 dt: float,
                 enable_plots: bool = False):
        self.providers   = providers
        self.subscribers = subscribers
        self.dt          = dt
        self.enable_plots= enable_plots

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
        except KeyboardInterrupt:
            log.info("Stopping …")
        # finally:
            # for s in self.subscribers: s.finish()
            # for p in self.providers:   p.stop()

    # ---------- optional ----------
    def run_calibration(self):
        log.info("Calibration routine (placeholder)")
        # your calibration steps here
