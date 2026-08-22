"""
Subsidence Analysis Module
==========================
Analytical and probabilistic modeling for production-induced surface subsidence.

Available Methods:
------------------
- Geertsma (1973): Deterministic subsidence for disk reservoirs
- Monte Carlo: Probabilistic uncertainty quantification
- Sensitivity Analysis: Tornado and Sobol methods
"""

# Deterministic Geertsma model
from .geertsma import (
    calculate_pore_compressibility,
    calculate_geertsma_factor,
    calculate_max_subsidence,
    calculate_subsidence_profile,
    fault_reactivation_risk,
)

# Monte Carlo & Sensitivity Analysis
from .monte_carlo import (
    MCParameter,
    MCConfig,
    MonteCarloSubsidence,
    scalar_subsidence,
    tornado_analysis,
    sobol_analysis,
    time_evolution,
)

__version__ = "1.1.0"
__author__ = "Saeed Gharedaghi"
__all__ = [
    # Geertsma
    "calculate_pore_compressibility", "calculate_geertsma_factor",
    "calculate_max_subsidence", "calculate_subsidence_profile",
    "fault_reactivation_risk",
    # Monte Carlo
    "MCParameter", "MCConfig", "MonteCarloSubsidence",
    "scalar_subsidence", "tornado_analysis", "sobol_analysis",
    "time_evolution",
]
