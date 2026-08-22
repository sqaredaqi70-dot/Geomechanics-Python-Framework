"""
Dynamic to Static Calibration Module (3D)
=========================================
Provides regression models (Linear and Power Law) to calibrate seismic-derived
dynamic elastic moduli to static lab/well properties.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional


def power_law_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = a * x^b."""
    return a * np.power(np.maximum(x, 0.01), b)


def calibrate_dynamic_to_static_e(
    e_dynamic_gpa: np.ndarray,
    e_static_gpa: np.ndarray
) -> Dict[str, Union[float, str]]:
    """
    Fits Linear and Power-Law regressions for Dynamic to Static Young's Modulus conversion.

    Parameters
    ----------
    e_dynamic_gpa : np.ndarray
        Dynamic Young's modulus from well/seismic in GPa.
    e_static_gpa : np.ndarray
        Static Young's modulus from lab cores/MEM in GPa.

    Returns
    -------
    Dict
        Best fitting model parameters, R-squared, and equation string.
    """
    mask = (e_dynamic_gpa > 1) & (e_dynamic_gpa < 200) & (e_static_gpa > 0.5) & (e_static_gpa < 150)
    ed = e_dynamic_gpa[mask]
    es = e_static_gpa[mask]

    if len(ed) < 10:
        # Fallback default if insufficient data
        return {"model": "fallback", "a": 0.5, "b": 1.0, "r2": 0.0, "eq": "E_stat = 0.5 * E_dyn"}

    # Linear Fit
    slope, intercept, r_val, _, _ = sp_stats.linregress(ed, es)
    r2_lin = r_val ** 2

    # Power Law Fit
    try:
        popt, _ = curve_fit(power_law_func, ed, es, p0=[1.0, 0.8], maxfev=5000, bounds=([0, 0], [100, 3]))
        es_pred = power_law_func(ed, *popt)
        r2_pow = sp_stats.pearsonr(es, es_pred)[0] ** 2
    except Exception:
        r2_pow = -1.0
        popt = [0.5, 1.0]

    if r2_pow > r2_lin:
        return {
            "model": "power",
            "a": popt[0],
            "b": popt[1],
            "r2": r2_pow,
            "eq": f"E_stat = {popt[0]:.4f} * E_dyn^{popt[1]:.4f}"
        }
    else:
        return {
            "model": "linear",
            "a": slope,
            "b": intercept,
            "r2": r2_lin,
            "eq": f"E_stat = {slope:.4f} * E_dyn + {intercept:.4f}"
        }


def calibrate_e_to_ucs(
    e_static_gpa: np.ndarray,
    ucs_mpa: np.ndarray
) -> Dict[str, Union[float, str]]:
    """
    Fits a Power Law model to convert Static Young's Modulus (GPa) to UCS (MPa).
    """
    mask = (e_static_gpa > 0.5) & (ucs_mpa > 1) & (ucs_mpa < 400)
    es = e_static_gpa[mask]
    ucs = ucs_mpa[mask]

    if len(es) < 10:
        return {"model": "fallback", "a": 2.5, "b": 0.9, "r2": 0.0, "eq": "UCS = 2.5 * E_stat^0.9"}

    try:
        popt, _ = curve_fit(power_law_func, es, ucs, p0=[5.0, 0.8], maxfev=5000, bounds=([0, 0], [100, 3]))
        ucs_pred = power_law_func(es, *popt)
        r2 = sp_stats.pearsonr(ucs, ucs_pred)[0] ** 2
        return {
            "model": "power",
            "a": popt[0],
            "b": popt[1],
            "r2": r2,
            "eq": f"UCS = {popt[0]:.4f} * E_stat^{popt[1]:.4f}"
        }
    except Exception:
        return {"model": "fallback", "a": 2.5, "b": 0.9, "r2": 0.0, "eq": "UCS = 2.5 * E_stat^0.9"}
