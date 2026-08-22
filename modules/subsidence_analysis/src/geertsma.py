"""
Geertsma Subsidence Model Module
================================
Analytical modeling of production-induced surface subsidence based on 
Geertsma (1973) formulation for poroelastic half-space.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
from typing import Union, Tuple, Optional

def calculate_pore_compressibility(e_modulus_gpa: float, poisson_ratio: float) -> float:
    """
    Calculates pore volume compressibility based on linear elasticity.

    Parameters
    ----------
    e_modulus_gpa : float
        Static Young's modulus of the reservoir rock in GPa.
    poisson_ratio : float
        Static Poisson's ratio of the reservoir rock.

    Returns
    -------
    float
        Pore compressibility in [1/MPa].
    """
    # Convert GPa to MPa for consistency (E * 1000)
    cm_per_mpa = (1.0 - 2.0 * poisson_ratio) * (1.0 + poisson_ratio) / (e_modulus_gpa * 1000.0 * (1.0 - poisson_ratio))
    return cm_per_mpa

def calculate_geertsma_factor(depth_m: float, radius_m: float) -> float:
    """
    Calculates the Geertsma geometry factor for a disk-shaped reservoir.
    
    Ref: uz(0) = alpha * Cm * dPp * h * 2*(1 - D/sqrt(D^2+R^2))

    Parameters
    ----------
    depth_m : float
        Mid-depth of the reservoir in meters.
    radius_m : float
        Equivalent disk radius of the reservoir in meters.

    Returns
    -------
    float
        Dimensionless geometry factor.
    """
    return 2.0 * (1.0 - depth_m / np.sqrt(depth_m**2 + radius_m**2))

def calculate_max_subsidence(
    depletion_mpa: Union[float, np.ndarray],
    h_eff_m: float,
    depth_m: float,
    radius_m: float,
    e_modulus_gpa: float,
    poisson_ratio: float,
    alpha_biot: float = 0.825
) -> Union[float, np.ndarray]:
    """
    Calculates the maximum surface subsidence at the center of the subsidence bowl.

    Parameters
    ----------
    depletion_mpa : float or np.ndarray
        Reservoir pore pressure depletion in MPa.
    h_eff_m : float
        Effective reservoir thickness in meters.
    depth_m : float
        Mid-depth of the reservoir in meters.
    radius_m : float
        Equivalent radius of the reservoir in meters.
    e_modulus_gpa : float
        Static Young's modulus in GPa.
    poisson_ratio : float
        Poisson's ratio.
    alpha_biot : float, optional
        Biot's poroelastic coefficient, by default 0.825.

    Returns
    -------
    float or np.ndarray
        Maximum surface subsidence in centimeters (cm).
    """
    cm_per_mpa = calculate_pore_compressibility(e_modulus_gpa, poisson_ratio)
    factor = calculate_geertsma_factor(depth_m, radius_m)
    
    # Calculate subsidence in meters, then convert to cm (* 100)
    uz_center_m = alpha_biot * cm_per_mpa * depletion_mpa * h_eff_m * factor
    return uz_center_m * 100.0

def calculate_subsidence_profile(
    r_distance_m: Union[float, np.ndarray],
    depletion_mpa: float,
    h_eff_m: float,
    depth_m: float,
    radius_m: float,
    e_modulus_gpa: float,
    poisson_ratio: float,
    alpha_biot: float = 0.825
) -> Union[float, np.ndarray]:
    """
    Calculates surface subsidence at a specific radial distance from the reservoir center.
    Uses the spatial decay function: uz(r) = uz(0) / (1+(r/R)^2)^0.5

    Parameters
    ----------
    r_distance_m : float or np.ndarray
        Radial distance from the center of the subsidence bowl in meters.
    (other parameters same as calculate_max_subsidence)

    Returns
    -------
    float or np.ndarray
        Subsidence at distance r in centimeters (cm).
    """
    uz_center_cm = calculate_max_subsidence(
        depletion_mpa, h_eff_m, depth_m, radius_m, 
        e_modulus_gpa, poisson_ratio, alpha_biot
    )
    
    uz_at_r_cm = uz_center_cm / (1.0 + (r_distance_m / radius_m)**2)**0.5
    return uz_at_r_cm

def fault_reactivation_risk(
    depletion_mpa: Union[float, np.ndarray],
    sh_max_eff: float,
    sh_min_eff: float,
    mu_fault: float = 0.769,
    fault_dip_deg: float = 65.0
) -> Union[float, np.ndarray]:
    """
    Calculates normalized shear stress (Ts_norm) on a fault to assess reactivation risk
    due to poroelastic stress changes from depletion.

    Parameters
    ----------
    depletion_mpa : float or np.ndarray
        Pore pressure drop in MPa.
    sh_max_eff : float
        Initial effective maximum horizontal stress in MPa.
    sh_min_eff : float
        Initial effective minimum horizontal stress in MPa.
    mu_fault : float, optional
        Friction coefficient of the fault, by default 0.769.
    fault_dip_deg : float, optional
        Dip angle of the fault in degrees, by default 65.0.

    Returns
    -------
    float or np.ndarray
        Normalized shear stress (Ts_norm). Values >= 1.0 indicate failure/reactivation.
    """
    # Assuming standard fault stress path behavior (delta S = delta Pp) for simplicity
    # Modify poroelastic path if required based on user's framework
    s1 = sh_max_eff - depletion_mpa
    s3 = sh_min_eff - depletion_mpa
    
    alpha = np.radians(fault_dip_deg)
    
    tau = 0.5 * np.abs(s1 - s3) * np.sin(2 * alpha)
    sn = 0.5 * (s1 + s3) + 0.5 * (s1 - s3) * np.cos(2 * alpha)
    
    # Handle negative normal stress (tension)
    ts_norm = np.where(sn > 0, tau / (mu_fault * sn), np.nan)
    return ts_norm
