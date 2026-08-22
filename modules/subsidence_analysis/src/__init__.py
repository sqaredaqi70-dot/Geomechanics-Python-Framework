"""
Subsidence Analysis Module
==========================
Comprehensive framework for production-induced surface subsidence modeling.

Available Methods:
------------------
1. Geertsma (1973): Deterministic analytical subsidence.
2. Monte Carlo: Probabilistic uncertainty & sensitivity (Tornado/Sobol).
3. Gibson / Terzaghi: Time-dependent consolidation.
4. DIF (Nucleus of Strain): Spatial integration using influence functions.
5. Comparator: Benchmarking tool for multi-model comparison.
6. Visualization: Publication-quality plotting utilities.

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
    MCParameter, MCConfig, MonteCarloSubsidence,
    scalar_subsidence, tornado_analysis, sobol_analysis, time_evolution,
)

# ── Gibson / Terzaghi Consolidation ───────────────────────────
from .gibson import (
    GibsonScenario, GibsonSubsidence, DEFAULT_SCENARIOS,
    terzaghi_degree_of_consolidation, compute_time_factor, get_drainage_path,
)

# ── Depth Influence Function (DIF) ────────────────────────────
from .influence_function import (
    nucleus_of_strain_kernel, compute_dif_subsidence,
)

# ── Multi-Model Comparator ────────────────────────────────────
from .comparator import SubsidenceComparator

# ── Visualization ─────────────────────────────────────────────
from .visualization import (
    apply_publication_style,
    plot_radial_profile,
    plot_subsidence_vs_depletion,
    plot_subsidence_map,
    plot_risk_zonation_map,
    plot_3d_deformed_surface,
    plot_mc_distributions,
    plot_tornado_sensitivity,
    plot_sobol_indices,
    plot_method_comparison,
    save_figure,
)

__version__ = "1.4.0"
__author__ = "Saeed Gharedaghi"

__all__ = [
    # Geertsma
    "calculate_pore_compressibility", "calculate_geertsma_factor",
    "calculate_max_subsidence", "calculate_subsidence_profile",
    "fault_reactivation_risk",
    # Monte Carlo
    "MCParameter", "MCConfig", "MonteCarloSubsidence", "scalar_subsidence",
    "tornado_analysis", "sobol_analysis", "time_evolution",
    # Gibson
    "GibsonScenario", "GibsonSubsidence", "DEFAULT_SCENARIOS",
    "terzaghi_degree_of_consolidation", "compute_time_factor", "get_drainage_path",
    # DIF
    "nucleus_of_strain_kernel", "compute_dif_subsidence",
    # Comparator
    "SubsidenceComparator",
    # Visualization
    "apply_publication_style", "plot_radial_profile", "plot_subsidence_vs_depletion",
    "plot_subsidence_map", "plot_risk_zonation_map", "plot_3d_deformed_surface",
    "plot_mc_distributions", "plot_tornado_sensitivity", "plot_sobol_indices",
    "plot_method_comparison", "save_figure",
]
