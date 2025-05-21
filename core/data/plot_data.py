from typing import Optional, Dict, Any, List
import numpy as np
from dataclasses import dataclass, field
from core.data.car_state import State, States


@dataclass
class PlotData:
    """
    Unified container for all visualizable data in PlotManager.
    Add any new fields you need later without breaking code.
    """
    state: Optional[State] = None
    states: Optional[States] = None
    path: Optional[np.ndarray] = None
    cones_map: Optional[Dict[str, np.ndarray]] = None
    cones_lidar: Optional[np.ndarray] = None
    cx: Optional[np.ndarray] = None
    cy: Optional[np.ndarray] = None
    target_ind: Optional[int] = None
    v_log: Optional[float] = None
    steering: Optional[float] = None
    acceleration: Optional[float] = None
    full_path: Optional[List[tuple]] = field(default_factory=list)
    X: Optional[List] = field(default_factory=list)
    Y: Optional[List] = field(default_factory=list)
    intermediate: Optional[Dict] = field(default_factory=dict)
