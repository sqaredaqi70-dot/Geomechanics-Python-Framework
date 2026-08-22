"""
3D Reservoir Geomechanics Module
================================
Integrated 3D geomechanical modeling driven by seismic inversion cubes and well data.

Features:
---------
- 3D Dynamic elastic moduli from Vp, Vs, Density
- Regression-based Dynamic to Static calibration
- 3D Static properties (E_static, PR_static, UCS, Cm)
- 3D Stress tensor computation (Sv, Pp, Shmin, SHmax)

Author: Saeed Gharedaghi
"""

from .calibration_3d import calibrate_dynamic_to_static_e, calibrate_e_to_ucs
from .elastic_properties_3d import compute_3d_dynamic_moduli, compute_3d_static_properties
from .stress_model_3d import compute_3d_stress_field

__version__ = "1.0.0"
__author__ = "Saeed Gharedaghi"

__all__ = [
    "calibrate_dynamic_to_static_e",
    "calibrate_e_to_ucs",
    "compute_3d_dynamic_moduli",
    "compute_3d_static_properties",
    "compute_3d_stress_field",
]
