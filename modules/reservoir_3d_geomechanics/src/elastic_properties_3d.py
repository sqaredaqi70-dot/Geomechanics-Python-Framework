"""
3D Elastic Properties Calculation Module
========================================
Computes 3D dynamic and static elastic moduli (Young's Modulus, Poisson's Ratio,
Shear Modulus, Bulk Modulus) from 3D seismic velocity (Vp, Vs) and density cubes.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
from typing import Dict, Tuple


def compute_3d_dynamic_moduli(
    vp_m_s: np.ndarray,
    vs_m_s: np.ndarray,
    density_g_cc: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes 3D dynamic elastic properties from Vp, Vs, and Density arrays.

    Parameters
    ----------
    vp_m_s : np.ndarray
        3D Array of P-wave velocity in m/s.
    vs_m_s : np.ndarray
        3D Array of S-wave velocity in m/s.
    density_g_cc : np.ndarray
        3D Array of Bulk Density in g/cm³.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing 3D arrays for 'pr_dyn', 'g_dyn_gpa', 'e_dyn_gpa', and 'k_dyn_gpa'.
    """
    rho_kg_m3 = density_g_cc * 1000.0  # Convert g/cc to kg/m³
    vp2 = vp_m_s ** 2
    vs2 = vs_m_s ** 2

    # Dynamic Poisson's Ratio
    pr_dyn = (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))
    pr_dyn = np.clip(pr_dyn, 0.05, 0.45)

    # Dynamic Shear Modulus G (GPa)
    g_dyn_gpa = (rho_kg_m3 * vs2) / 1e9
    g_dyn_gpa = np.clip(g_dyn_gpa, 1.0, 80.0)

    # Dynamic Young's Modulus E (GPa)
    e_dyn_gpa = 2.0 * g_dyn_gpa * (1.0 + pr_dyn)
    e_dyn_gpa = np.clip(e_dyn_gpa, 2.0, 200.0)

    # Dynamic Bulk Modulus K (GPa)
    k_dyn_gpa = rho_kg_m3 * (vp2 - (4.0 / 3.0) * vs2) / 1e9
    k_dyn_gpa = np.clip(k_dyn_gpa, 2.0, 150.0)

    return {
        "pr_dyn": pr_dyn,
        "g_dyn_gpa": g_dyn_gpa,
        "e_dyn_gpa": e_dyn_gpa,
        "k_dyn_gpa": k_dyn_gpa,
    }


def compute_3d_static_properties(
    e_dyn_3d: np.ndarray,
    pr_dyn_3d: np.ndarray,
    e_calib_params: Dict,
    ucs_calib_params: Dict,
    scale_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D dynamic properties into static E, static Poisson's ratio, UCS, and Cm.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (e_stat_gpa, pr_stat, ucs_mpa, cm_per_mpa)
    """
    # Apply calibration
    if e_calib_params.get("model") == "power":
        a, b = e_calib_params["a"], e_calib_params["b"]
        e_stat_gpa = a * np.power(np.maximum(e_dyn_3d, 0.01), b)
    else:
        a, b = e_calib_params.get("a", 0.5), e_calib_params.get("b", 1.0)
        e_stat_gpa = a * e_dyn_3d + b

    # Apply Well-Scale Factor Correction if provided
    e_stat_gpa = np.clip(e_stat_gpa * scale_factor, 1.0, 100.0)

    # Static Poisson's ratio (typically identical to dynamic in carbonates)
    pr_stat = np.clip(pr_dyn_3d, 0.10, 0.45)

    # UCS Calculation
    a_u, b_u = ucs_calib_params.get("a", 2.5), ucs_calib_params.get("b", 0.9)
    ucs_mpa = a_u * np.power(np.maximum(e_stat_gpa, 0.01), b_u)
    ucs_mpa = np.clip(ucs_mpa, 1.0, 500.0)

    # Pore Compressibility Cm [1/MPa]
    e_mpa = e_stat_gpa * 1000.0
    cm_per_mpa = ((1.0 + pr_stat) * (1.0 - 2.0 * pr_stat)) / (e_mpa * (1.0 - pr_stat))
    cm_per_mpa = np.clip(cm_per_mpa, 1e-6, 5e-4)

    return e_stat_gpa, pr_stat, ucs_mpa, cm_per_mpa
