"""
3D Wellbore Stability Module
"""

from .kirsch_3d import (
    transform_stresses_to_wellbore,
    compute_kirsch_stresses,
    calculate_mud_weight_window,
)

__version__ = "1.0.0"
__author__ = "Saeed Gharedaghi"

__all__ = [
    "transform_stresses_to_wellbore",
    "compute_kirsch_stresses",
    "calculate_mud_weight_window",
]
