"""
Monte Carlo Subsidence Analysis Module
=======================================
Probabilistic subsidence assessment based on Geertsma (1973) with 
uncertainty quantification, sensitivity analysis (Tornado & Sobol),
and time-evolution modeling.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from .geertsma import calculate_pore_compressibility


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════
@dataclass
class MCParameter:
    """
    Definition of a single Monte Carlo input parameter with truncated Normal distribution.
    
    Parameters
    ----------
    mean : float
        Mean value of the distribution.
    std : float
        Standard deviation.
    min_val : float
        Lower truncation bound.
    max_val : float
        Upper truncation bound.
    """
    mean: float
    std: float
    min_val: float
    max_val: float


@dataclass
class MCConfig:
    """Container for full Monte Carlo simulation configuration."""
    E_static: MCParameter = field(
        default_factory=lambda: MCParameter(27.75, 4.28, 10.0, 80.0))
    poisson_ratio: MCParameter = field(
        default_factory=lambda: MCParameter(0.228, 0.030, 0.10, 0.40))
    ntg: MCParameter = field(
        default_factory=lambda: MCParameter(0.473, 0.021, 0.20, 0.80))
    h_gross: MCParameter = field(
        default_factory=lambda: MCParameter(223.0, 30.0, 100.0, 400.0))
    radius: MCParameter = field(
        default_factory=lambda: MCParameter(5000.0, 500.0, 3000.0, 8000.0))
    depth: MCParameter = field(
        default_factory=lambda: MCParameter(2300.0, 100.0, 1800.0, 2800.0))
    alpha_biot_min: float = 0.70
    alpha_biot_max: float = 0.95
    n_iterations: int = 10_000
    random_seed: int = 42


# ══════════════════════════════════════════════════════════════
# CORE MC ENGINE
# ══════════════════════════════════════════════════════════════
class MonteCarloSubsidence:
    """
    Monte Carlo simulator for surface subsidence with uncertainty quantification.
    
    Implements the Geertsma (1973) analytical model over Monte Carlo samples
    to derive probability distributions of surface subsidence.
    """
    
    def __init__(self, config: Optional[MCConfig] = None):
        """
        Initialize the Monte Carlo engine.

        Parameters
        ----------
        config : MCConfig, optional
            Simulation configuration. If None, default values are used.
        """
        self.config = config if config is not None else MCConfig()
        self.samples: Dict[str, np.ndarray] = {}
        self.results: Dict[float, Dict] = {}
        
    def _sample_truncated_normal(self, param: MCParameter, n: int) -> np.ndarray:
        """Generate truncated normal samples for a given parameter."""
        arr = np.random.normal(param.mean, param.std, n)
        return np.clip(arr, param.min_val, param.max_val)
    
    def generate_samples(self) -> Dict[str, np.ndarray]:
        """
        Generate all Monte Carlo input samples with truncated distributions.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing sample arrays for each input parameter.
        """
        np.random.seed(self.config.random_seed)
        n = self.config.n_iterations
        
        self.samples = {
            "E_static":   self._sample_truncated_normal(self.config.E_static, n),
            "poisson":    self._sample_truncated_normal(self.config.poisson_ratio, n),
            "ntg":        self._sample_truncated_normal(self.config.ntg, n),
            "h_gross":    self._sample_truncated_normal(self.config.h_gross, n),
            "radius":     self._sample_truncated_normal(self.config.radius, n),
            "depth":      self._sample_truncated_normal(self.config.depth, n),
            "alpha_biot": np.random.uniform(
                self.config.alpha_biot_min, 
                self.config.alpha_biot_max, n),
        }
        
        # Derived properties
        self.samples["Cm"] = calculate_pore_compressibility(
            self.samples["E_static"], self.samples["poisson"])
        self.samples["h_eff"] = self.samples["h_gross"] * self.samples["ntg"]
        
        return self.samples
    
    def _geertsma_center_vectorized(
        self, 
        depletion_mpa: float,
        cm_arr: np.ndarray,
        h_eff_arr: np.ndarray,
        alpha_arr: np.ndarray,
        radius_arr: np.ndarray,
        depth_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vectorized subsidence at reservoir center over MC samples.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (surface_subsidence_m, reservoir_compaction_m)
        """
        compaction = alpha_arr * cm_arr * depletion_mpa * h_eff_arr
        factor = 2.0 * (1.0 - depth_arr / np.sqrt(depth_arr**2 + radius_arr**2))
        return compaction * factor, compaction
    
    def run_scenario(self, depletion_mpa: float) -> Dict:
        """
        Run MC simulation for a single depletion scenario.

        Parameters
        ----------
        depletion_mpa : float
            Reservoir pressure depletion in MPa.

        Returns
        -------
        Dict
            Full statistical summary including percentiles P5, P10, P25, P50, P75, P90, P95.
        """
        if not self.samples:
            self.generate_samples()
        
        sub_m, comp_m = self._geertsma_center_vectorized(
            depletion_mpa,
            self.samples["Cm"], self.samples["h_eff"],
            self.samples["alpha_biot"], self.samples["radius"],
            self.samples["depth"]
        )
        sub_cm = sub_m * 100
        comp_cm = comp_m * 100
        
        stats = {
            "sub_samples_cm": sub_cm,
            "comp_samples_cm": comp_cm,
            "mean": float(np.mean(sub_cm)),
            "std":  float(np.std(sub_cm)),
            "min":  float(np.min(sub_cm)),
            "max":  float(np.max(sub_cm)),
        }
        for p in [5, 10, 25, 50, 75, 90, 95]:
            stats[f"P{p}"] = float(np.percentile(sub_cm, p))
        
        return stats
    
    def run_all_scenarios(
        self, depletion_scenarios_mpa: List[float]
    ) -> Dict[float, Dict]:
        """
        Run MC simulation for multiple depletion scenarios.

        Parameters
        ----------
        depletion_scenarios_mpa : List[float]
            List of depletion values (MPa) to simulate.

        Returns
        -------
        Dict[float, Dict]
            Nested dictionary indexed by depletion value.
        """
        self.results = {
            dpp: self.run_scenario(dpp) for dpp in depletion_scenarios_mpa
        }
        return self.results
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return MC results as a summary DataFrame."""
        rows = []
        for dpp, r in self.results.items():
            rows.append({
                "dPp_MPa": dpp,
                "mean_cm": round(r["mean"], 4),
                "std_cm":  round(r["std"], 4),
                "P5":      round(r["P5"], 4),
                "P50":     round(r["P50"], 4),
                "P95":     round(r["P95"], 4),
                "min":     round(r["min"], 4),
                "max":     round(r["max"], 4),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════
def scalar_subsidence(
    e_gpa, poisson, ntg, h_gross, radius, depth, alpha, depletion_mpa=10.0
):
    """
    Scalar subsidence function for sensitivity analyses (Tornado, Sobol).
    All arguments positional to allow vectorization over samples.

    Returns
    -------
    float or np.ndarray
        Subsidence in cm.
    """
    cm = calculate_pore_compressibility(e_gpa, poisson)
    h_eff = h_gross * ntg
    compaction = alpha * cm * depletion_mpa * h_eff
    factor = 2.0 * (1.0 - depth / np.sqrt(depth**2 + radius**2))
    return compaction * factor * 100.0  # cm


def tornado_analysis(config: MCConfig, depletion_mpa: float = 10.0) -> pd.DataFrame:
    """
    Compute Tornado sensitivity (±1σ variation from mean).

    Parameters
    ----------
    config : MCConfig
        Monte Carlo configuration containing parameter distributions.
    depletion_mpa : float
        Depletion at which sensitivity is evaluated.

    Returns
    -------
    pd.DataFrame
        Sorted Tornado results with base, low, high, and swing values.
    """
    alpha_mean = (config.alpha_biot_min + config.alpha_biot_max) / 2.0
    
    means = {
        "E_static":   config.E_static.mean,
        "poisson":    config.poisson_ratio.mean,
        "ntg":        config.ntg.mean,
        "h_gross":    config.h_gross.mean,
        "radius":     config.radius.mean,
        "depth":      config.depth.mean,
    }
    base = scalar_subsidence(**means, alpha=alpha_mean, depletion_mpa=depletion_mpa)
    
    param_map = {
        "E_static": config.E_static,   "poisson": config.poisson_ratio,
        "ntg":      config.ntg,        "h_gross": config.h_gross,
        "radius":   config.radius,     "depth":   config.depth,
    }
    
    rows = []
    for pk, param in param_map.items():
        lo_vals = dict(means); hi_vals = dict(means)
        lo_vals[pk] = param.mean - param.std
        hi_vals[pk] = param.mean + param.std
        
        s_lo = scalar_subsidence(**lo_vals, alpha=alpha_mean, depletion_mpa=depletion_mpa)
        s_hi = scalar_subsidence(**hi_vals, alpha=alpha_mean, depletion_mpa=depletion_mpa)
        
        rows.append({
            "parameter": pk, "base_cm": round(base, 4),
            "low_cm":    round(s_lo, 4), "high_cm": round(s_hi, 4),
            "swing_cm":  round(abs(s_hi - s_lo), 4),
        })
    
    # Alpha Biot swing (uniform distribution)
    s_al = scalar_subsidence(**means, alpha=config.alpha_biot_min, depletion_mpa=depletion_mpa)
    s_ah = scalar_subsidence(**means, alpha=config.alpha_biot_max, depletion_mpa=depletion_mpa)
    rows.append({
        "parameter": "alpha_biot", "base_cm": round(base, 4),
        "low_cm":    round(s_al, 4), "high_cm": round(s_ah, 4),
        "swing_cm":  round(abs(s_ah - s_al), 4),
    })
    
    return pd.DataFrame(rows).sort_values("swing_cm", ascending=False)


def sobol_analysis(
    config: MCConfig, 
    depletion_mpa: float = 10.0,
    n_samples: int = 2048,
    seed: int = 789
) -> pd.DataFrame:
    """
    Perform Sobol variance-based sensitivity analysis (S1 and ST indices).

    Parameters
    ----------
    config : MCConfig
        MC configuration.
    depletion_mpa : float
        Depletion scenario at which indices are computed.
    n_samples : int
        Base sample size for Sobol matrices.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sobol first-order (S1) and total-order (ST) indices per parameter.
    """
    np.random.seed(seed)
    
    param_order = ["E_static", "poisson", "ntg", "h_gross", "radius", "depth", "alpha_biot"]
    
    def sample_col(pname, n):
        if pname == "alpha_biot":
            return np.random.uniform(config.alpha_biot_min, config.alpha_biot_max, n)
        p = getattr(config, {"E_static":"E_static", "poisson":"poisson_ratio",
                             "ntg":"ntg", "h_gross":"h_gross",
                             "radius":"radius", "depth":"depth"}[pname])
        return np.clip(np.random.normal(p.mean, p.std, n), p.min_val, p.max_val)
    
    A = np.column_stack([sample_col(p, n_samples) for p in param_order])
    B = np.column_stack([sample_col(p, n_samples) for p in param_order])
    
    def eval_matrix(M):
        return scalar_subsidence(
            M[:,0], M[:,1], M[:,2], M[:,3], M[:,4], M[:,5], M[:,6],
            depletion_mpa=depletion_mpa
        )
    
    YA, YB = eval_matrix(A), eval_matrix(B)
    var_total = np.var(np.concatenate([YA, YB]))
    
    rows = []
    for i, p in enumerate(param_order):
        AB = A.copy(); AB[:, i] = B[:, i]
        YAB = eval_matrix(AB)
        S1 = max(0, np.mean(YB * (YAB - YA)) / var_total)
        ST = max(0, 0.5 * np.mean((YA - YAB)**2) / var_total)
        rows.append({"parameter": p, "S1": round(S1, 4), "ST": round(ST, 4)})
    
    return pd.DataFrame(rows).sort_values("ST", ascending=False)


# ══════════════════════════════════════════════════════════════
# TIME EVOLUTION
# ══════════════════════════════════════════════════════════════
def time_evolution(
    mc_engine: MonteCarloSubsidence,
    years: np.ndarray,
    depletion_rate_mpa_per_year: float = 2.0
) -> pd.DataFrame:
    """
    Compute subsidence evolution over time given a constant depletion rate.

    Parameters
    ----------
    mc_engine : MonteCarloSubsidence
        Initialized MC engine (samples will be generated if not already).
    years : np.ndarray
        Array of years to evaluate.
    depletion_rate_mpa_per_year : float
        Reservoir depletion rate (MPa/year).

    Returns
    -------
    pd.DataFrame
        Time evolution with mean, P5, and P95 subsidence.
    """
    if not mc_engine.samples:
        mc_engine.generate_samples()
    
    rows = []
    for t in years:
        dpp_t = t * depletion_rate_mpa_per_year
        if dpp_t == 0:
            rows.append({"year": t, "dPp_MPa": 0.0, 
                        "mean_cm": 0.0, "P5_cm": 0.0, "P95_cm": 0.0})
        else:
            stats = mc_engine.run_scenario(dpp_t)
            rows.append({
                "year": t, "dPp_MPa": dpp_t,
                "mean_cm": round(stats["mean"], 4),
                "P5_cm":   round(stats["P5"], 4),
                "P95_cm":  round(stats["P95"], 4),
            })
    return pd.DataFrame(rows)
