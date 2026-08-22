"""
3D In-Situ Stress Calculation Module
====================================
Computes 3D Overburden Stress (Sv), Pore Pressure (Pp), Minimum Horizontal Stress (Shmin),
and Maximum Horizontal Stress (SHmax) fields across the 3D reservoir grid.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
from typing import Dict, Tuple


def compute_3d_stress_field(
    density_g_cc_3d: np.ndarray,
    depth_m_3d: np.ndarray,
    pr_stat_3d: np.ndarray,
    pore_pressure_gradient_m_pa_m: float = 0.0113,  # ~0.5 psi/ft
    tectonic_stress_mpa: float = 0.0,
    sh_sh_ratio: float = 1.15,
    g_ms2: float = 9.81
) -> Dict[str, np.ndarray]:
    """
    Computes full 3D stress tensors using density-integrated Sv and poroelastic Shmin/SHmax.

    Parameters
    ----------
    density_g_cc_3d : np.ndarray
        3D bulk density in g/cm³.
    depth_m_3d : np.ndarray
        3D True Vertical Depth array (m).
    pr_stat_3d : np.ndarray
        3D Static Poisson's ratio.
    pore_pressure_gradient_m_pa_m : float
        Pore pressure gradient in MPa/m.
    tectonic_stress_mpa : float
        Additional tectonic stress component in MPa.
    sh_sh_ratio : float
        Ratio of SHmax to Shmin (from image logs or leak-off tests).

    Returns
    -------
    Dict[str, np.ndarray]
        3D Stress components: 'sv_mpa', 'pp_mpa', 'shmin_mpa', 'shmax_mpa'.
    """
    rho_kg_m3 = density_g_cc_3d * 1000.0

    # Overburden Stress Sv (MPa) = rho * g * depth / 1e6
    sv_mpa = (rho_kg_m3 * g_ms2 * np.abs(depth_m_3d)) / 1e6
    sv_mpa = np.clip(sv_mpa, 1.0, 250.0)

    # Pore Pressure Pp (MPa)
    pp_mpa = np.abs(depth_m_3d) * pore_pressure_gradient_m_pa_m
    pp_mpa = np.clip(pp_mpa, 1.0, sv_mpa * 0.95)

    # Effective Horizontal Stress factor (Poroelastic formulation)
    poro_factor = pr_stat_3d / (1.0 - pr_stat_3d)

    # Minimum Horizontal Stress Shmin (MPa)
    shmin_mpa = (poro_factor * (sv_mpa - pp_mpa)) + pp_mpa + tectonic_stress_mpa
    shmin_mpa = np.clip(shmin_mpa, pp_mpa, sv_mpa * 1.5)

    # Maximum Horizontal Stress SHmax (MPa)
    shmax_mpa = shmin_mpa * sh_sh_ratio
    shmax_mpa = np.clip(shmax_mpa, shmin_mpa, sv_mpa * 2.0)

    return {
        "sv_mpa": sv_mpa,
        "pp_mpa": pp_mpa,
        "shmin_mpa": shmin_mpa,
        "shmax_mpa": shmax_mpa
    }
