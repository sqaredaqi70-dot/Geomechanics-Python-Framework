"""
═══════════════════════════════════════════════════════════════════════
                    GEOMECHANICS MODULE
═══════════════════════════════════════════════════════════════════════

Core calculations for the 1D Mechanical Earth Model, including:
- Vertical stress from density integration
- Stress gradients
- Elastic moduli conversion
- Wellbore stability (collapse, fracture, mud weight)
- Stress regime classification

Author:  Saeed Gharedaghi
Contact: sqaredaqi70@gmail.com
License: AGPL-3.0 (Academic) / Commercial (see LICENSE files)
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd

from . import config as cfg


# UTILITY FUNCTIONS
def safe_median(series):
    """Compute median safely, handling NaN and inf."""
    if series is None:
        return np.nan
    v = pd.to_numeric(series, errors='coerce')
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.median()) if len(v) > 0 else np.nan


def pressure_to_gradient_gcc(p_mpa, depth_m):
    """Convert pressure (MPa) to equivalent density (g/cc)."""
    p = np.asarray(p_mpa, dtype=float)
    d = np.asarray(depth_m, dtype=float)
    d = np.where(d > 0, d, np.nan)
    return p * 1000.0 / (cfg.G_MS2 * d)


def pressure_to_gradient_ppg(p_mpa, depth_m):
    """Convert pressure (MPa) to equivalent mud weight (ppg)."""
    return pressure_to_gradient_gcc(p_mpa, depth_m) * cfg.GCC_TO_PPG


# VERTICAL STRESS FROM DENSITY INTEGRATION
def integrate_vertical_stress(depth_m, density_gcc):
    """
    Compute vertical stress by integrating density with depth.

    Sv(z) = ∫₀ᶻ ρ(z') · g · dz'

    Parameters
    ----------
    depth_m : array
        Depth values (meters).
    density_gcc : array
        Bulk density (g/cc).

    Returns
    -------
    np.array
        Vertical stress in MPa.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    rho_kgm3 = np.asarray(density_gcc, dtype=float) * 1000.0

    sv_pa = np.zeros_like(depth_m)
    for k in range(1, len(depth_m)):
        dz = depth_m[k] - depth_m[k - 1]
        if dz > 0:
            sv_pa[k] = (sv_pa[k - 1] +
                        0.5 * (rho_kgm3[k] + rho_kgm3[k - 1]) *
                        cfg.G_MS2 * dz)

    return sv_pa / 1e6  # Pa → MPa


# STRESS GRADIENT COMPUTATION
def compute_stress_gradients(df):
    """
    Compute stress gradients (psi/ft) from stress columns (psi).

    Adds columns: Sv_grad, SH_grad, Sh_grad, Pp_grad, Pp_ppg.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'DEPTH_ft' and stress columns in psi.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame.
    """
    df = df.copy()

    for psi_col, grad_col in [
        ('Sv_psi', 'Sv_grad'),
        ('SH_psi', 'SH_grad'),
        ('Sh_psi', 'Sh_grad'),
        ('Pp_psi', 'Pp_grad'),
    ]:
        if psi_col in df.columns and 'DEPTH_ft' in df.columns:
            df[grad_col] = np.where(
                df['DEPTH_ft'] > 0,
                df[psi_col] / df['DEPTH_ft'],
                np.nan
            )

    if 'Pp_grad' in df.columns:
        df['Pp_ppg'] = df['Pp_grad'] / cfg.PPG_TO_PSIFT

    return df


# ELASTIC MODULI CONVERSION
def convert_moduli_to_mpsi(df):
    """
    Convert elastic moduli from GPa to Mpsi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with moduli columns in GPa.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with additional _Mpsi columns.
    """
    df = df.copy()

    conversions = [
        ('YME_DYN',      'E_dyn_Mpsi'),
        ('E_FINAL',      'E_stat_Mpsi'),
        ('SMG_DYN',      'G_dyn_Mpsi'),
        ('BMK_DYN',      'K_dyn_Mpsi'),
        ('YME_STA_HMC',  'E_HMC_Mpsi'),
        ('YME_STA_JFC',  'E_JFC_Mpsi'),
        ('YME_STA_MMC',  'E_MMC_Mpsi'),
        ('YME_STA_PBC',  'E_PBC_Mpsi'),
    ]

    for src, dst in conversions:
        if src in df.columns:
            df[dst] = df[src] * cfg.GPA_TO_MPSI

    return df


# WELLBORE STABILITY (MUD WEIGHT WINDOW)
def compute_mud_weight_window(df):
    """
    Compute collapse, fracture, and moderated collapse mud weights.

    Uses Kirsch equations with Mohr-Coulomb failure criterion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SHMAX_PHS, SHMIN_PHS, FINAL_PP, UCS_FINAL,
        FANG_FROMGR, and DEPTH columns.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with mud weight columns.
    """
    df = df.copy()

    required = ['SHMAX_PHS', 'SHMIN_PHS', 'FINAL_PP',
                'UCS_FINAL', 'FANG_FROMGR', 'DEPTH']
    if not all(c in df.columns for c in required):
        return df

    SH  = df['SHMAX_PHS'].values.astype(float)
    Sh  = df['SHMIN_PHS'].values.astype(float)
    Pp  = df['FINAL_PP'].values.astype(float)
    UCS = df['UCS_FINAL'].values.astype(float)
    phi = np.radians(df['FANG_FROMGR'].values.astype(float))
    dep = df['DEPTH'].values.astype(float)

    # Mohr-Coulomb passive coefficient
    q = (1 + np.sin(phi)) / (1 - np.sin(phi) + 1e-9)

    # Tensile strength
    T0 = UCS * cfg.T0_FRACTION

    # Collapse mud pressure (MPa)
    Pc = (3 * SH - Sh - UCS + Pp * (q - 1)) / (q + 1)

    # Fracture mud pressure (MPa)
    Pf = 3 * Sh - SH + T0 - Pp

    # Convert to ppg
    df['MW_col_ppg'] = pressure_to_gradient_ppg(Pc, dep)
    df['MW_frc_ppg'] = pressure_to_gradient_ppg(Pf, dep)
    df['MW_pp_ppg']  = pressure_to_gradient_ppg(Pp, dep)

    if 'Sv_MPa' in df.columns:
        df['MW_sv_ppg'] = pressure_to_gradient_ppg(
            df['Sv_MPa'].values, dep
        )

    # Window width
    df['MW_win_ppg'] = df['MW_frc_ppg'] - df['MW_col_ppg']

    # Moderated collapse (with safety margin)
    df['MW_moderated_collapse_ppg'] = (
        df['MW_col_ppg'] + cfg.MW_SAFETY_MARGIN
    )

    return df



# STRESS REGIME CLASSIFICATION
def classify_stress_regime(df):
    """
    Classify stress regime per depth sample.

    Adds 'REGIME' column with values: NF, SS, or RF.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Sv_MPa, SHMAX_PHS, SHMIN_PHS columns.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame.
    """
    df = df.copy()

    if not all(c in df.columns for c in
                ['Sv_MPa', 'SHMAX_PHS', 'SHMIN_PHS']):
        return df

    Sv = df['Sv_MPa'].values
    SH = df['SHMAX_PHS'].values
    Sh = df['SHMIN_PHS'].values

    df['REGIME'] = np.where(
        (SH > Sv) & (Sv > Sh),
        'SS',
        np.where(Sv > SH, 'NF', 'RF')
    )

    return df


def determine_overall_regime(wells):
    """
    Determine overall stress regime across all wells.

    Parameters
    ----------
    wells : dict
        Dictionary of well data.

    Returns
    -------
    dict
        Contains: 'K0', 'kH', 'regime_str', 'regime_code'.
    """
    sh_sv_ratios = []
    sH_sv_ratios = []

    for wa, wd in wells.items():
        df = wd['df']
        sv = df['Sv_MPa'].replace(0, np.nan) if 'Sv_MPa' in df.columns else None
        if sv is None:
            continue

        if 'SHMIN_PHS' in df.columns:
            r = (df['SHMIN_PHS'] / sv).dropna()
            sh_sv_ratios.extend(r[(r > 0) & (r < 5)].tolist())

        if 'SHMAX_PHS' in df.columns:
            r = (df['SHMAX_PHS'] / sv).dropna()
            sH_sv_ratios.extend(r[(r > 0) & (r < 5)].tolist())

    K0 = float(np.median(sh_sv_ratios)) if sh_sv_ratios else np.nan
    kH = float(np.median(sH_sv_ratios)) if sH_sv_ratios else np.nan

    if kH > 1.0 and K0 < 1.0:
        regime_str  = "Compressional with strike-slip component"
        regime_code = "SS-C"
    elif kH > 1.0 and K0 > 1.0:
        regime_str  = "Reverse Faulting"
        regime_code = "RF"
    elif kH < 1.0 and K0 < 1.0:
        regime_str  = "Normal Faulting"
        regime_code = "NF"
    else:
        regime_str  = "Mixed"
        regime_code = "MIX"

    return {
        'K0':          K0,
        'kH':          kH,
        'regime_str':  regime_str,
        'regime_code': regime_code,
    }


# COMPLETE MEM PROCESSING (MAIN FUNCTION)
def process_well(well_data):
    """
    Apply full MEM processing pipeline to one well.

    Parameters
    ----------
    well_data : dict
        Well data dictionary from data_loader.

    Returns
    -------
    dict
        Well data with processed DataFrame.
    """
    df = well_data['df']

    # Compute Sv from density if available
    if 'RHOB' in df.columns and 'Sv_MPa' not in df.columns:
        # Use density to compute Sv
        df_sorted = df.sort_values('DEPTH').reset_index(drop=True)
        sv_mpa = integrate_vertical_stress(
            df_sorted['DEPTH'].values,
            df_sorted['RHOB'].fillna(cfg.RHO_FALLBACK).values
        )
        df_sorted['Sv_MPa'] = sv_mpa
        df = df_sorted

    # Fallback: lithostatic gradient
    if 'Sv_MPa' not in df.columns:
        df['Sv_MPa'] = (cfg.RHO_FALLBACK * cfg.G_MS2 *
                        df['DEPTH'] / 1000.0)

    df['Sv_psi'] = df['Sv_MPa'] * cfg.MPA_TO_PSI

    # Apply all transformations
    df = compute_stress_gradients(df)
    df = convert_moduli_to_mpsi(df)
    df = compute_mud_weight_window(df)
    df = classify_stress_regime(df)

    # UCS in MPa (short name)
    if 'UCS_FINAL' in df.columns:
        df['UCS_MPa'] = df['UCS_FINAL']

    well_data['df'] = df
    return well_data
