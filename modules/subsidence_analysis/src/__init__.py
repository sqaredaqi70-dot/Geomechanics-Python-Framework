"""
Subsidence Analysis Module
==========================
Geertsma (1973) based analytical modeling for production-induced subsidence.
"""

from .geertsma import (
    calculate_pore_compressibility,
    calculate_geertsma_factor,
    calculate_max_subsidence,
    calculate_subsidence_profile,
    fault_reactivation_risk,
)

__version__ = "1.0.0"
__author__ = "Saeed Gharedaghi"
