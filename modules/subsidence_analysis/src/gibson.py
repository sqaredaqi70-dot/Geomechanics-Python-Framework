"""
Gibson / Terzaghi Consolidation Subsidence Module
==================================================
Time-dependent subsidence based on 1D consolidation theory (Terzaghi, 1943;
Gibson, 1958) coupled with Geertsma poroelastic framework.

The module implements:
- Degree of consolidation U(t) via Terzaghi series solution
- Single and double drainage conditions
- Coefficient of consolidation (Cv) sensitivity scenarios
- Time-series subsidence evolution
- Coupled Geertsma-Gibson comparison

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from .geertsma import calculate_pore_compressibility, calculate_geertsma_factor


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════
@dataclass
class GibsonScenario:
    """
    Configuration for a single Gibson/Terzaghi consolidation scenario.

    Parameters
    ----------
    name : str
        Identifier for the scenario.
    cv_m2_per_s : float
        Coefficient of consolidation (m²/s). Typical range: 1e-8 to 1e-4.
    drainage : str
        "single" (one-way drainage, Hdr = H) or "double" (Hdr = H/2).
    description : str, optional
        Human-readable description.
    """
    name: str
    cv_m2_per_s: float
    drainage: str = "single"
    description: str = ""

    def __post_init__(self):
        if self.drainage.lower() not in ("single", "double"):
            raise ValueError(f"drainage must be 'single' or 'double', got {self.drainage}")


# Default set of consolidation scenarios (literature-based typical values)
DEFAULT_SCENARIOS = [
    GibsonScenario("LOW_Cv_single",  1e-8, "single", "Low Cv - slow consolidation"),
    GibsonScenario("BASE_Cv_single", 1e-6, "single", "Base case - typical carbonate"),
    GibsonScenario("HIGH_Cv_single", 1e-4, "single", "High Cv - fast consolidation"),
    GibsonScenario("BASE_Cv_double", 1e-6, "double", "Base case with double drainage"),
]


# ══════════════════════════════════════════════════════════════
# CORE CONSOLIDATION FUNCTIONS
# ══════════════════════════════════════════════════════════════
def terzaghi_degree_of_consolidation(
    time_factor: Union[float, np.ndarray],
    n_terms: int = 50
) -> Union[float, np.ndarray]:
    """
    Compute the degree of consolidation U(Tv) using Terzaghi's series solution.

    Formula:
        U(Tv) = 1 - Σ [8 / (π²·(2m+1)²)] · exp[-(2m+1)²·π²·Tv / 4]

    Parameters
    ----------
    time_factor : float or np.ndarray
        Dimensionless time factor Tv = Cv·t / Hdr²
    n_terms : int, optional
        Number of series terms for convergence (default 50).

    Returns
    -------
    float or np.ndarray
        Degree of consolidation in range [0, 1].
    """
    tv = np.asarray(time_factor, dtype=float)
    tv = np.maximum(tv, 0.0)
    s = np.zeros_like(tv, dtype=float)
    for m in range(n_terms):
        k = 2 * m + 1
        s += 8.0 / (np.pi**2 * k**2) * np.exp(-(k**2) * (np.pi**2) * tv / 4.0)
    return np.clip(1.0 - s, 0.0, 1.0)


def compute_time_factor(
    cv_m2_per_s: float,
    time_seconds: Union[float, np.ndarray],
    drainage_path_m: float
) -> Union[float, np.ndarray]:
    """
    Compute the dimensionless time factor Tv.

    Parameters
    ----------
    cv_m2_per_s : float
        Coefficient of consolidation in m²/s.
    time_seconds : float or np.ndarray
        Time elapsed in seconds.
    drainage_path_m : float
        Longest drainage path in meters (Hdr).
        For single drainage: Hdr = H (full thickness)
        For double drainage: Hdr = H / 2

    Returns
    -------
    float or np.ndarray
        Dimensionless time factor.
    """
    return cv_m2_per_s * np.asarray(time_seconds) / max(drainage_path_m**2, 1e-12)


def get_drainage_path(thickness_m: float, drainage: str) -> float:
    """
    Return effective drainage path length based on drainage condition.

    Parameters
    ----------
    thickness_m : float
        Reservoir thickness in meters.
    drainage : str
        "single" or "double".

    Returns
    -------
    float
        Effective drainage path length.
    """
    return thickness_m / 2.0 if drainage.lower() == "double" else thickness_m


# ══════════════════════════════════════════════════════════════
# GIBSON SUBSIDENCE CALCULATOR
# ══════════════════════════════════════════════════════════════
class GibsonSubsidence:
    """
    Time-dependent subsidence calculator using Gibson/Terzaghi consolidation
    coupled with Geertsma poroelastic framework.
    """

    def __init__(
        self,
        h_eff_m: float,
        depth_m: float,
        radius_m: float,
        e_modulus_gpa: float,
        poisson_ratio: float,
        alpha_biot: float = 0.825
    ):
        """
        Initialize the Gibson subsidence calculator.

        Parameters
        ----------
        h_eff_m : float
            Effective reservoir thickness in meters.
        depth_m : float
            Mid-depth of the reservoir in meters.
        radius_m : float
            Equivalent disk radius in meters.
        e_modulus_gpa : float
            Static Young's modulus in GPa.
        poisson_ratio : float
            Static Poisson's ratio.
        alpha_biot : float
            Biot poroelastic coefficient (default 0.825).
        """
        self.h_eff_m = h_eff_m
        self.depth_m = depth_m
        self.radius_m = radius_m
        self.e_modulus_gpa = e_modulus_gpa
        self.poisson_ratio = poisson_ratio
        self.alpha_biot = alpha_biot

        # Pre-computed constants
        self.cm = calculate_pore_compressibility(e_modulus_gpa, poisson_ratio)
        self.geertsma_factor = calculate_geertsma_factor(depth_m, radius_m)

    def geertsma_max_subsidence(self, depletion_mpa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Instantaneous (elastic) Geertsma subsidence in cm."""
        uz_center_m = self.alpha_biot * self.cm * depletion_mpa * self.h_eff_m * self.geertsma_factor
        return uz_center_m * 100.0

    def gibson_max_subsidence(
        self,
        depletion_mpa: Union[float, np.ndarray],
        time_seconds: Union[float, np.ndarray],
        cv_m2_per_s: float,
        drainage: str = "single"
    ) -> Union[float, np.ndarray]:
        """
        Time-dependent Gibson subsidence (elastic × degree of consolidation).

        Parameters
        ----------
        depletion_mpa : float or np.ndarray
            Reservoir depletion in MPa.
        time_seconds : float or np.ndarray
            Time elapsed in seconds.
        cv_m2_per_s : float
            Coefficient of consolidation in m²/s.
        drainage : str
            "single" or "double" drainage condition.

        Returns
        -------
        float or np.ndarray
            Gibson subsidence in cm.
        """
        h_drain = get_drainage_path(self.h_eff_m, drainage)
        tv = compute_time_factor(cv_m2_per_s, time_seconds, h_drain)
        u = terzaghi_degree_of_consolidation(tv)
        return self.geertsma_max_subsidence(depletion_mpa) * u

    def run_time_series(
        self,
        df_depletion: pd.DataFrame,
        scenario: GibsonScenario,
        depletion_col: str = "depletion_mpa",
        time_col: str = "time_seconds"
    ) -> pd.DataFrame:
        """
        Compute full time-series subsidence for a given scenario.

        Parameters
        ----------
        df_depletion : pd.DataFrame
            Time-series data with depletion and time columns.
        scenario : GibsonScenario
            Consolidation scenario to apply.
        depletion_col, time_col : str
            Column names for depletion (MPa) and elapsed time (seconds).

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame with U, Geertsma, and Gibson columns.
        """
        out = df_depletion.copy()
        h_drain = get_drainage_path(self.h_eff_m, scenario.drainage)
        tv = compute_time_factor(scenario.cv_m2_per_s, out[time_col].values, h_drain)
        u = terzaghi_degree_of_consolidation(tv)

        geertsma_cm = self.geertsma_max_subsidence(out[depletion_col].values)
        gibson_cm = geertsma_cm * u

        out[f"{scenario.name}_U"] = u
        out[f"{scenario.name}_geertsma_cm"] = geertsma_cm
        out[f"{scenario.name}_gibson_cm"] = gibson_cm
        return out

    def run_all_scenarios(
        self,
        df_depletion: pd.DataFrame,
        scenarios: Optional[List[GibsonScenario]] = None,
        depletion_col: str = "depletion_mpa",
        time_col: str = "time_seconds"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all Gibson scenarios and return time-series + summary tables.

        Parameters
        ----------
        df_depletion : pd.DataFrame
            Time-series data with depletion and time columns.
        scenarios : List[GibsonScenario], optional
            List of scenarios. If None, uses DEFAULT_SCENARIOS.
        depletion_col, time_col : str
            Column names.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (time_series, summary_stats)
        """
        if scenarios is None:
            scenarios = DEFAULT_SCENARIOS

        ts = df_depletion.copy()
        geertsma_peak = self.geertsma_max_subsidence(ts[depletion_col].max())

        rows = []
        for sc in scenarios:
            ts = self.run_time_series(ts, sc, depletion_col, time_col)
            gib_peak = float(ts[f"{sc.name}_gibson_cm"].max())
            u_peak = float(ts[f"{sc.name}_U"].max())
            ratio = gib_peak / geertsma_peak if geertsma_peak > 0 else np.nan
            rows.append({
                "scenario": sc.name,
                "cv_m2_per_s": sc.cv_m2_per_s,
                "drainage": sc.drainage,
                "h_drainage_m": get_drainage_path(self.h_eff_m, sc.drainage),
                "U_peak": round(u_peak, 4),
                "peak_gibson_cm": round(gib_peak, 4),
                "geertsma_peak_cm": round(float(geertsma_peak), 4),
                "gibson_to_geertsma_ratio": round(ratio, 4),
            })

        summary = pd.DataFrame(rows)
        return ts, summary
