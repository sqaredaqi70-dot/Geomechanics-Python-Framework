"""
Subsidence Analysis Module
==========================
Comprehensive framework for production-induced surface subsidence modeling.

Available Methods:
------------------
1. Geertsma (1973):
   Deterministic analytical subsidence for disk-shaped reservoirs.

2. Monte Carlo:
   Probabilistic uncertainty quantification with truncated distributions,
   Tornado sensitivity, and Sobol variance-based analysis.

3. Gibson / Terzaghi:
   Time-dependent consolidation coupled with Geertsma poroelastic framework.
   Includes degree of consolidation U(t), single/double drainage, and
   coefficient of consolidation (Cv) scenarios.

Author: Saeed Gharedaghi
"""

# ── Deterministic Geertsma model ──────────────────────────────
from .geertsma import (
    calculate_pore_compressibility,
    calculate_geertsma_factor,
    calculate_max_subsidence,
    calculate_subsidence_profile,
    fault_reactivation_risk,
)

# ── Monte Carlo & Sensitivity Analysis ────────────────────────
from .monte_carlo import (
    MCParameter,
    MCConfig,
    MonteCarloSubsidence,
    scalar_subsidence,
    tornado_analysis,
    sobol_analysis,
    time_evolution,
)

# ── Gibson / Terzaghi Consolidation ───────────────────────────
from .gibson import (
    GibsonScenario,
    GibsonSubsidence,
    DEFAULT_SCENARIOS,
    terzaghi_degree_of_consolidation,
    compute_time_factor,
    get_drainage_path,
)

__version__ = "1.2.0"
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
    # Gibson / Terzaghi
    "GibsonScenario", "GibsonSubsidence", "DEFAULT_SCENARIOS",
    "terzaghi_degree_of_consolidation", "compute_time_factor",
    "get_drainage_path",
]
