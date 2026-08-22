"""
3D Wellbore Stability Module (Kirsch Equations & Mud Weight Window)
====================================================================
Computes 3D stress concentration around inclined wellbores using Kirsch equations,
and evaluates Mud Weight Windows (MWW) to prevent Shear Breakouts and Tensile Fractures.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
from typing import Dict, Tuple


def transform_stresses_to_wellbore(
    sv_mpa: float,
    shmax_mpa: float,
    shmin_mpa: float,
    pp_mpa: float,
    shmax_azimuth_deg: float,
    well_inclination_deg: float,
    well_azimuth_deg: float
) -> Dict[str, float]:
    """
    Transforms far-field stress tensor (Sv, SHmax, Shmin) to the local wellbore coordinate system.
    """
    inc = np.radians(well_inclination_deg)
    azi = np.radians(well_azimuth_deg)
    s_az = np.radians(shmax_azimuth_deg)

    # Relative azimuth between well and SHmax
    gamma = azi - s_az

    # Transformation matrix components
    l1 = np.sin(inc) * np.cos(gamma)
    l2 = np.sin(inc) * np.sin(gamma)
    l3 = np.cos(inc)

    m1 = np.cos(inc) * np.cos(gamma)
    m2 = np.cos(inc) * np.sin(gamma)
    m3 = -np.sin(inc)

    n1 = -np.sin(gamma)
    n2 = np.cos(gamma)
    n3 = 0.0

    # In-situ stress tensor in geographic frame
    s_geo = np.diag([shmax_mpa, shmin_mpa, sv_mpa])
    R = np.array([[l1, l2, l3], [m1, m2, m3], [n1, n2, n3]])

    # Transformed stress tensor in wellbore local frame (x_w, y_w, z_w)
    s_well = R @ s_geo @ R.T

    return {
        "s_xx": s_well[0, 0],
        "s_yy": s_well[1, 1],
        "s_zz": s_well[2, 2],
        "t_xy": s_well[0, 1],
        "t_yz": s_well[1, 2],
        "t_xz": s_well[0, 2],
        "pp": pp_mpa
    }


def compute_kirsch_stresses(
    well_stresses: Dict[str, float],
    p_mud_mpa: float,
    theta_deg: np.ndarray,
    poisson_ratio: float = 0.25
) -> Dict[str, np.ndarray]:
    """
    Calculates Kirsch effective stress components at the wellbore wall (r = R_well).
    """
    th = np.radians(theta_deg)
    s_xx = well_stresses["s_xx"]
    s_yy = well_stresses["s_yy"]
    s_zz = well_stresses["s_zz"]
    t_xy = well_stresses["t_xy"]
    t_yz = well_stresses["t_yz"]
    t_xz = well_stresses["t_xz"]
    pp = well_stresses["pp"]

    # Effective mud pressure at wellbore wall
    p_eff = p_mud_mpa - pp

    # Effective Kirsch stress components at the wall
    s_theta = (s_xx + s_yy - 2.0 * p_eff) - 2.0 * (s_xx - s_yy) * np.cos(2.0 * th) - 4.0 * t_xy * np.sin(2.0 * th)
    s_z = s_zz - pp - poisson_ratio * (2.0 * (s_xx - s_yy) * np.cos(2.0 * th) + 4.0 * t_xy * np.sin(2.0 * th))
    t_theta_z = 2.0 * (t_yz * np.cos(th) - t_xz * np.sin(th))

    # Principal stresses at wellbore wall
    sigma_1 = 0.5 * (s_theta + s_z) + 0.5 * np.sqrt((s_theta - s_z)**2 + 4.0 * t_theta_z**2)
    sigma_3 = 0.5 * (s_theta + s_z) - 0.5 * np.sqrt((s_theta - s_z)**2 + 4.0 * t_theta_z**2)
    sigma_r = p_eff  # Radial stress equals effective mud pressure

    return {
        "sigma_1": sigma_1,
        "sigma_3": sigma_3,
        "sigma_r": np.full_like(sigma_1, sigma_r),
        "sigma_theta": s_theta
    }


def calculate_mud_weight_window(
    depth_m: float,
    sv_mpa: float,
    shmax_mpa: float,
    shmin_mpa: float,
    pp_mpa: float,
    ucs_mpa: float,
    friction_angle_deg: float = 30.0,
    tensile_strength_mpa: float = 0.0,
    shmax_azimuth_deg: float = 0.0,
    inclination_deg: float = 0.0,
    azimuth_deg: float = 0.0
) -> Dict[str, float]:
    """
    Computes minimum mud weight to prevent Shear Breakout (P_breakout)
    and maximum mud weight to prevent Tensile Fracture (P_tensile) in g/cm³ (SG).
    """
    g_ms2 = 9.81
    m_pa_to_sg = 1e6 / (g_ms2 * depth_m * 1000.0)  # Convert MPa to g/cm3 (SG)

    well_stresses = transform_stresses_to_wellbore(
        sv_mpa, shmax_mpa, shmin_mpa, pp_mpa,
        shmax_azimuth_deg, inclination_deg, azimuth_deg
    )

    theta = np.linspace(0, 360, 360)
    phi_rad = np.radians(friction_angle_deg)
    q_f = (1.0 + np.sin(phi_rad)) / (1.0 - np.sin(phi_rad))

    # Estimate Breakout pressure (Min Mud Weight) via Mohr-Coulomb
    p_mud_range = np.linspace(pp_mpa * 0.5, sv_mpa * 1.5, 200)
    p_bo = pp_mpa

    for pm in p_mud_range:
        stresses = compute_kirsch_stresses(well_stresses, pm, theta)
        s1 = np.maximum(stresses["sigma_1"], stresses["sigma_theta"])
        s3 = np.minimum(stresses["sigma_3"], stresses["sigma_r"])
        # Check Mohr-Coulomb failure criterion
        failure = (s1 - s3 * q_f) - ucs_mpa
        if np.any(failure > 0):
            p_bo = pm  # Required mud pressure to suppress breakout
        else:
            break

    # Estimate Tensile Fracture pressure (Max Mud Weight)
    p_tf = sv_mpa * 1.5
    for pm in reversed(p_mud_range):
        stresses = compute_kirsch_stresses(well_stresses, pm, theta)
        min_theta_stress = np.min(stresses["sigma_theta"])
        if min_theta_stress < -tensile_strength_mpa:
            p_tf = pm

    return {
        "pore_pressure_sg": pp_mpa * m_pa_to_sg,
        "breakout_mw_sg": p_bo * m_pa_to_sg,
        "tensile_mw_sg": p_tf * m_pa_to_sg,
        "overburden_sg": sv_mpa * m_pa_to_sg,
    }
