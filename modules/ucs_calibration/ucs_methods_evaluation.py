#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCS Methods Evaluation — Zone-Specific Calibration for Heterogeneous Carbonates
───────────────────────────────────────────────────────────────────────────────
Evaluates 11 published UCS correlations against measured UCS and derives 3 new
power-law calibrations (global-static, global-dynamic, zone-specific) with
honest cross-validation and leave-one-well-out metrics.

INPUT   : master_dataset.csv  (DEPTH, Well, ZONE, UCS_FINAL, E_FINAL, YME_DYN, ...)
OUTPUT  : Fig01..Fig07, Table5/7/9 CSVs, figures_summary.json, PAPER_REPORT.txt

RULES
-----
- Zero hardcoded placeholder metrics; every number traces back to the input CSV.
- Negative R² values are preserved (strict formulation, no clipping at zero).
- CV / LOWO refit model coefficients inside each fold (no train-test contamination).
- If input CSV is absent, a synthetic Fallback is generated for pipeline testing
  only — flagged clearly in logs so it cannot be mistaken for real results.
"""

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Reservoir zone taxonomy and color scheme (publication-consistent)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_ORDER = ["Zone-1", "Zone-2A", "Zone-2B", "Zone-3"]
ZCOL = {
    "Zone-1":  "#9467bd",
    "Zone-2A": "#e377c2",
    "Zone-2B": "#8c564b",
    "Zone-3":  "#d62728",
    "Undiff.": "gray",
}
WCOL = {"Well-A": "#1f77b4", "Well-B": "#ff7f0e", "Well-C": "#2ca02c"}

FIELD_NAME  = "a representative carbonate reservoir"
FIELD_SHORT = "Reservoir-X"

# Font hierarchy — titles > panel > labels > ticks > annotations
TYPE = {
    "main":   24.0,
    "panel":  15.0,
    "axes":   15.5,
    "ticks":  13.5,
    "annot":  12.0,
    "legend": 12.0,
    "cb":     13.0,
    "cell":   11.0,
}
PANEL_SIZE = TYPE["panel"]

# UCS method registry
METHOD_META = {
    "UCS_HORSRUD":       dict(label="Horsrud (2001)",                color="#1f77b4", litho="Shale/Chalk",   family="published", fitted=False),
    "UCS_MCNALLY":       dict(label="McNally (1987)",                color="#ff7f0e", litho="Coal measures", family="published", fitted=False),
    "UCS_CDE":           dict(label="CDE / Chang (2006)",            color="#2ca02c", litho="General",       family="vendor",    fitted=False),
    "UCS_SMG_RPC":       dict(label="Shear-modulus (RPC)",           color="#d62728", litho="Various",       family="vendor",    fitted=False),
    "UCS_SND_RPC":       dict(label="Sonic-based (RPC)",             color="#9467bd", litho="Various",       family="vendor",    fitted=False),
    "UCS_YME":           dict(label="E-based (RPC)",                 color="#8c564b", litho="Various",       family="vendor",    fitted=False),
    "Militzer_1973":     dict(label="Militzer & Stoll (1973)",       color="#17becf", litho="Limestone",     family="published", fitted=False),
    "Golubev_1976":      dict(label="Golubev & Rabinovich (1976)",   color="#bcbd22", litho="Carbonate",     family="published", fitted=False),
    "Chang_2006":        dict(label="Chang et al. (2006) carbonate", color="#e377c2", litho="Carbonate",     family="published", fitted=False),
    "Lacy_1997":         dict(label="Lacy (1997)",                   color="#7f7f7f", litho="General",       family="published", fitted=False),
    "Bradford_1998":     dict(label="Bradford et al. (1998)",        color="#aec7e8", litho="Carbonate",     family="published", fitted=False),
    "NEW_ZoneSpec":      dict(label="Zone-specific PL (this study)",       color="#2ecc71", litho="Target carbonate", family="new", fitted=True),
    "NEW_GlobalStat_PL": dict(label="Global static-E PL (this study)",     color="#e74c3c", litho="Target carbonate", family="new", fitted=True),
    "NEW_GlobalDyn_PL":  dict(label="Global dynamic-E PL (this study)",    color="#f39c12", litho="Target carbonate", family="new", fitted=True),
}

ALIASES = {
    "Militzer_1973":  ["Militzer_1973", "Militzer(1973)"],
    "Golubev_1976":   ["Golubev_1976", "Golubev(1976)"],
    "Chang_2006":     ["Chang_2006", "Chang(2006)_Carb"],
    "Lacy_1997":      ["Lacy_1997", "Lacy(1997)"],
    "Bradford_1998":  ["Bradford_1998", "Bradford(1998)"],
}

PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
    "figure.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "axes.titleweight": "bold", "axes.labelweight": "bold",
    "figure.dpi": 100, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": ":",
    "lines.linewidth": 2.0, "axes.linewidth": 1.5,
}

LOG_BUFFER = []

def log(msg=""):
    print(msg, flush=True)
    LOG_BUFFER.append(str(msg))

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────
def power_law(x, a, b):
    return a * np.power(np.maximum(np.asarray(x, dtype=float), 0.01), b)

def fit_power_law(x, y):
    p, _ = curve_fit(power_law, x, y, p0=[5.0, 0.8], maxfev=5000,
                     bounds=([0, 0], [100, 3]))
    return float(p[0]), float(p[1])

def ccc_score(y_true, y_pred):
    """Lin's concordance correlation coefficient."""
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(ddof=1), yp.var(ddof=1)
    cov = np.cov(yt, yp, ddof=1)[0, 1]
    denom = vt + vp + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")

def strict_metrics(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    r_p = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else float("nan")
    return {
        "R2":    float(r2_score(yt, yp)),
        "RMSE":  float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":   float(mean_absolute_error(yt, yp)),
        "Bias":  float(np.mean(yp - yt)),
        "Pearson_r": r_p,
        "CCC":   ccc_score(yt, yp),
        "n":     int(len(yt)),
    }

def safe_boxplot(ax, data, labels=None, **kwargs):
    import matplotlib as mpl
    ver = tuple(int(x) for x in mpl.__version__.split(".")[:2])
    if labels is not None:
        if ver >= (3, 9): kwargs["tick_labels"] = labels
        else:             kwargs["labels"] = labels
    return ax.boxplot(data, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Fallback — only when the real CSV is absent (clearly flagged)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_master(dest: Path, n_rows=2400):
    """Generate placeholder CSV so the pipeline can be test-run on GitHub.
    This output must NEVER be used for scientific interpretation."""
    rng = np.random.RandomState(42)
    wells = ["Well-A", "Well-B", "Well-C"]
    rows = []
    for w in wells:
        depth = np.linspace(2000, 4000, n_rows // 3)
        for d in depth:
            if   d < 2400: z = "Zone-1"
            elif d < 2900: z = "Zone-2A"
            elif d < 3400: z = "Zone-2B"
            else:          z = "Zone-3"
            e_sta = max(1.0, rng.normal(28, 10))
            e_dyn = e_sta * rng.uniform(1.1, 1.6)
            # Zone-dependent power-law scatter around true UCS
            if   z == "Zone-1":  ucs = 7.5 * e_sta**0.80 + rng.normal(0, 8)
            elif z == "Zone-2A": ucs = 6.2 * e_sta**0.85 + rng.normal(0, 10)
            elif z == "Zone-2B": ucs = 5.8 * e_sta**0.90 + rng.normal(0, 12)
            else:                ucs = 9.1 * e_sta**0.75 + rng.normal(0, 7)
            ucs = max(5.0, ucs)
            rows.append({
                "DEPTH": d, "Well": w, "ZONE": z,
                "UCS_FINAL":  ucs,
                "E_FINAL":    e_sta,
                "YME_DYN":    e_dyn,
                "DEPTH_NORM": (d - 2000) / 2000,
                "UCS_HORSRUD":   0.77 * (305 / (e_dyn * 2))**2.93,
                "UCS_MCNALLY":   1.2e-7 * (305 / (e_dyn * 2))**3.0,
                "UCS_CDE":       7.22 * e_dyn**0.712,
                "UCS_SMG_RPC":   5.0 * e_sta**0.70 + rng.normal(0, 15),
                "UCS_SND_RPC":   6.5 * e_dyn**0.65 + rng.normal(0, 18),
                "UCS_YME":       0.9 * e_sta + rng.normal(0, 10),
                "Militzer_1973": 2.922 * e_dyn**0.960,
                "Golubev_1976":  0.0736 * e_dyn**1.673,
                "Chang_2006":    13.8 * e_dyn**0.51,
                "Lacy_1997":     0.278 * e_dyn**1.549,
                "Bradford_1998": 7.22 * e_dyn**0.712,
            })
    df = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log(f"  [SYNTHETIC] Fallback dataset generated at: {dest}")
    log(f"  [SYNTHETIC] This dataset is for pipeline testing only; results are not scientific.")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading and method resolution
# ─────────────────────────────────────────────────────────────────────────────
def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["DEPTH", "Well", "ZONE", "UCS_FINAL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] master CSV missing columns: {missing}")
    df["ZONE"] = df["ZONE"].fillna("Undiff.")
    return df

def resolve_columns(df: pd.DataFrame):
    colmap = {}
    for key in METHOD_META:
        for c in ALIASES.get(key, [key]):
            if c in df.columns:
                colmap[key] = c
                break
    return colmap

def build_new_methods(df: pd.DataFrame, colmap: dict):
    info = {}
    fit = df.dropna(subset=["E_FINAL", "YME_DYN", "UCS_FINAL", "ZONE"]).copy()
    fit = fit[(fit["E_FINAL"] > 0) & (fit["YME_DYN"] > 0) &
              (fit["UCS_FINAL"] > 0) & (fit["UCS_FINAL"] < 300) &
              (fit["E_FINAL"] < 150) & (fit["YME_DYN"] < 150)]
    if len(fit) < 50:
        raise SystemExit("[FATAL] insufficient rows (E_FINAL/YME_DYN/UCS) for fitting")

    a, b = fit_power_law(fit["E_FINAL"].values, fit["UCS_FINAL"].values)
    df["NEW_GlobalStat_PL"] = power_law(df["E_FINAL"].clip(0.01, 150), a, b)
    info["NEW_GlobalStat_PL"] = {"input": "E_FINAL", "a": a, "b": b,
                                 "eq": f"UCS = {a:.2f} * E_s^{b:.3f}", "n": len(fit)}

    a2, b2 = fit_power_law(fit["YME_DYN"].values, fit["UCS_FINAL"].values)
    df["NEW_GlobalDyn_PL"] = power_law(df["YME_DYN"].clip(0.01, 150), a2, b2)
    info["NEW_GlobalDyn_PL"] = {"input": "YME_DYN", "a": a2, "b": b2,
                                "eq": f"UCS = {a2:.2f} * E_d^{b2:.3f}", "n": len(fit)}

    df["NEW_ZoneSpec"] = np.nan
    zone_eqs = {}
    for z in ZONE_ORDER:
        zd = fit[fit["ZONE"] == z]
        if len(zd) < 20:
            continue
        az, bz = fit_power_law(zd["E_FINAL"].values, zd["UCS_FINAL"].values)
        m = df["ZONE"] == z
        df.loc[m, "NEW_ZoneSpec"] = power_law(df.loc[m, "E_FINAL"].clip(0.01, 150), az, bz)
        zone_eqs[z] = {"a": az, "b": bz,
                       "eq": f"UCS = {az:.2f} * E_s^{bz:.3f}", "n": int(len(zd))}
    info["NEW_ZoneSpec"] = {"input": "E_FINAL", "zones": zone_eqs}

    colmap["NEW_ZoneSpec"]      = "NEW_ZoneSpec"
    colmap["NEW_GlobalStat_PL"] = "NEW_GlobalStat_PL"
    colmap["NEW_GlobalDyn_PL"]  = "NEW_GlobalDyn_PL"
    return info

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(df, colmap):
    rows, zone_r2_store = [], {}
    for key, col in colmap.items():
        meta = METHOD_META[key]
        v = df[["UCS_FINAL", col, "ZONE"]].dropna()
        v = v[(v["UCS_FINAL"] > 0) & (v["UCS_FINAL"] < 300) &
              (v[col] > 0) & (v[col] < 500)]
        if len(v) < 20:
            log(f"  [skip] {key}: only {len(v)} valid rows")
            continue
        m = strict_metrics(v["UCS_FINAL"].values, v[col].values)
        m.update({"Key": key, "Method": meta["label"], "color": meta["color"],
                  "litho": meta["litho"], "family": meta["family"],
                  "fitted": meta["fitted"], "column": col})
        zr = {}
        for z in ZONE_ORDER:
            vz = v[v["ZONE"] == z]
            if len(vz) >= 10:
                zr[z] = float(r2_score(vz["UCS_FINAL"].values, vz[col].values))
        m["zone_R2"] = zr
        zone_r2_store[key] = zr
        rows.append(m)
    res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    res["Rank"] = res.index + 1
    return res, zone_r2_store

def cv_lowo_fitted(df, info):
    """CV + LOWO for fitted methods only; refit coefficients inside each fold."""
    out = {}
    fd = df.dropna(subset=["E_FINAL", "YME_DYN", "UCS_FINAL", "ZONE", "Well"]).copy()
    fd = fd[(fd["UCS_FINAL"] > 0) & (fd["UCS_FINAL"] < 300) &
            (fd["E_FINAL"] > 0) & (fd["E_FINAL"] < 150)]

    def _fit_global(train, col):
        return fit_power_law(train[col].values, train["UCS_FINAL"].values)

    def _pred_global(model, test, col):
        a, b = model
        return power_law(test[col].clip(0.01, 150).values, a, b)

    def _fit_zone(train):
        eqs = {}
        for z in ZONE_ORDER:
            d = train[train["ZONE"] == z]
            if len(d) >= 20:
                eqs[z] = fit_power_law(d["E_FINAL"].values, d["UCS_FINAL"].values)
        return eqs

    def _pred_zone(eqs, test):
        yp = np.full(len(test), np.nan)
        for z, (a, b) in eqs.items():
            m = (test["ZONE"] == z).values
            yp[m] = power_law(test.loc[m, "E_FINAL"].clip(0.01, 150).values, a, b)
        return yp

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, col, zone_flag in [
        ("NEW_GlobalStat_PL", "E_FINAL", False),
        ("NEW_GlobalDyn_PL",  "YME_DYN", False),
        ("NEW_ZoneSpec",      "E_FINAL", True),
    ]:
        oof = np.full(len(fd), np.nan)
        for tr_i, te_i in kf.split(fd):
            tr, te = fd.iloc[tr_i], fd.iloc[te_i]
            oof[te_i] = (_pred_zone(_fit_zone(tr), te) if zone_flag
                         else _pred_global(_fit_global(tr, col), te, col))
        ok = ~np.isnan(oof)
        if ok.sum() > 50:
            cv_m = strict_metrics(fd["UCS_FINAL"].values[ok], oof[ok])
            out[name] = {"CV_R2": cv_m["R2"], "CV_RMSE": cv_m["RMSE"], "cv_n": cv_m["n"]}

    wells = sorted(fd["Well"].unique())
    if len(wells) >= 3:
        for name, col, zone_flag in [
            ("NEW_GlobalStat_PL", "E_FINAL", False),
            ("NEW_GlobalDyn_PL",  "YME_DYN", False),
            ("NEW_ZoneSpec",      "E_FINAL", True),
        ]:
            oof = np.full(len(fd), np.nan)
            for w in wells:
                te_mask = (fd["Well"] == w).values
                tr, te = fd[~te_mask], fd[te_mask]
                if len(te) < 20:
                    continue
                oof[te_mask] = (_pred_zone(_fit_zone(tr), te) if zone_flag
                                else _pred_global(_fit_global(tr, col), te, col))
            ok = ~np.isnan(oof)
            if ok.sum() > 50:
                lo = strict_metrics(fd["UCS_FINAL"].values[ok], oof[ok])
                out.setdefault(name, {}).update(
                    {"LOWO_R2": lo["R2"], "LOWO_RMSE": lo["RMSE"], "lowo_n": lo["n"]})
    return out

def bootstrap_block(df, info, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    fd = df.dropna(subset=["E_FINAL", "UCS_FINAL", "ZONE"]).copy()
    fd = fd[(fd["UCS_FINAL"] > 0) & (fd["UCS_FINAL"] < 300) &
            (fd["E_FINAL"] > 0) & (fd["E_FINAL"] < 150)]
    X = fd["E_FINAL"].values; Y = fd["UCS_FINAL"].values

    boot_r2, boot_b = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        try:
            a, b = fit_power_law(X[idx], Y[idx])
            boot_r2.append(float(r2_score(Y, power_law(X, a, b))))
            boot_b.append(b)
        except Exception:
            pass
    boot_r2 = np.array(boot_r2); boot_b = np.array(boot_b)

    out = {
        "_boot_r2_array": boot_r2,
        "global_static": {
            "n_iter": int(len(boot_r2)),
            "mean":   float(boot_r2.mean()) if len(boot_r2) else None,
            "std":    float(boot_r2.std()) if len(boot_r2) else None,
            "ci95_lower": float(np.percentile(boot_r2, 2.5)) if len(boot_r2) else None,
            "ci95_upper": float(np.percentile(boot_r2, 97.5)) if len(boot_r2) else None,
        },
        "global_static_exponent": {
            "mean": float(boot_b.mean()) if len(boot_b) else None,
            "ci95": [float(np.percentile(boot_b, 2.5)), float(np.percentile(boot_b, 97.5))]
                    if len(boot_b) else None,
        },
        "zone_exponents": {},
    }
    for z in ZONE_ORDER:
        zd = fd[fd["ZONE"] == z]
        if len(zd) < 20:
            continue
        Xz, Yz = zd["E_FINAL"].values, zd["UCS_FINAL"].values
        bs = []
        for _ in range(n_boot):
            idx = rng.choice(len(Xz), size=len(Xz), replace=True)
            try:
                _, bz = fit_power_law(Xz[idx], Yz[idx])
                bs.append(bz)
            except Exception:
                pass
        bs = np.array(bs)
        if len(bs):
            out["zone_exponents"][z] = {
                "b_mean": float(bs.mean()),
                "b_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n_iter": int(len(bs)),
            }
    return out

def zone_stats_table(df):
    rows = []
    for z in ZONE_ORDER:
        d = df[df["ZONE"] == z]["UCS_FINAL"].dropna()
        d = d[(d > 0) & (d < 300)]
        if len(d) < 10:
            continue
        rows.append({
            "Zone": z, "n": int(len(d)),
            "Mean": float(d.mean()), "Std": float(d.std()),
            "P10":  float(d.quantile(0.10)),
            "Median": float(d.median()),
            "P90":  float(d.quantile(0.90)),
            "CV_pct": float(100 * d.std() / d.mean()) if d.mean() else np.nan,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, outdir, name, dpi):
    fp = outdir / name
    fig.savefig(fp, dpi=dpi)
    plt.close(fig)
    log(f"  ✓ {name}  ({fp.stat().st_size/1024:.0f} KB)")
    return name

def fig01_workflow(df, res, fit_info, outdir, dpi):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")

    def box(x, y, w, h, text, color, fs=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    linewidth=2, edgecolor="black",
                                    facecolor=color, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="black")

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                     mutation_scale=20, linewidth=2, color="black"))

    nw = df["Well"].nunique()
    n_tot = int(len(df))
    box(0.3, 9.4, 3.0, 1.1, f"LAS files\n{nw} wells", "#3498db", 11)
    box(3.7, 9.4, 3.0, 1.1, "Well tops\n(reservoir zones)", "#3498db", 11)
    box(7.1, 9.4, 2.6, 1.1, "Zone assignment\n(4 zones)", "#3498db", 11)
    box(2.0, 7.6, 6.0, 1.1, f"MASTER DATASET\nn = {n_tot:,} depth-matched pairs", "#1abc9c", 13)
    box(0.3, 5.5, 3.0, 1.6, "6 LAS methods\n(Horsrud, McNally,\nCDE, SMG, SND, YME)", "#f39c12", 11)
    box(3.7, 5.5, 3.0, 1.6, "5 Literature methods\n(Militzer, Golubev,\nChang, Lacy, Bradford)", "#f39c12", 11)
    box(7.1, 5.5, 2.6, 1.6, "3 NEW calibrations\n(Zone-specific,\nGlobal-Es, Global-Ed)", "#2ecc71", 11)
    box(1.0, 3.7, 8.0, 1.1, "7-METRIC EVALUATION\nR² (strict) · RMSE · MAE · Bias · CCC · CV-R² · Zone-R²", "#9b59b6", 12)
    box(0.5, 2.0, 4.2, 1.3, f"Uncertainty analysis\n{globals().get('N_BOOT', 1000):,} bootstrap iterations", "#e67e22", 11)
    box(5.3, 2.0, 4.2, 1.3, "Zone-disaggregated\nR² across 4 zones", "#e67e22", 11)
    box(1.0, 0.2, 8.0, 1.3, "FINAL RESULTS\n7 figures  ·  tables  ·  JSON summary", "#e74c3c", 13)

    for x in [1.8, 5.2, 8.4]:
        arr(x, 9.4, 5.0, 8.75)
    arr(5.0, 7.6, 5.0, 7.15)
    for x in [1.8, 5.2, 8.4]:
        arr(x, 5.5, 5.0, 4.85)
    arr(5.0, 3.7, 2.6, 3.3); arr(5.0, 3.7, 7.4, 3.3)
    arr(2.6, 2.0, 5.0, 1.5); arr(7.4, 2.0, 5.0, 1.5)

    ax.set_title(f"Fig. 1  Analytical workflow of UCS method evaluation\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, outdir, "Fig01_workflow.png", dpi)

def fig02_overview(df, res, fit_info, outdir, dpi):
    fig = plt.figure(figsize=(22, 7))
    gs = GridSpec(1, 4, wspace=0.35)
    wells = sorted(df["Well"].unique())

    ax = fig.add_subplot(gs[0])
    for w in wells:
        d = df[df["Well"] == w]["UCS_FINAL"].dropna()
        d = d[(d > 0) & (d < 300)]
        ax.hist(d, bins=40, alpha=0.60, color=WCOL.get(w, "gray"),
                label=w, density=True, orientation="horizontal")
    ax.set_ylabel("UCS (MPa)"); ax.set_xlabel("Density")
    ax.set_title("(a) Per-well distribution", fontsize=PANEL_SIZE, fontweight="bold")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[1])
    zone_order = [z for z in ZONE_ORDER if (df["ZONE"] == z).sum() >= 10]
    data = [df[(df["ZONE"] == z)]["UCS_FINAL"].dropna().values for z in zone_order]
    bp = safe_boxplot(ax, data, labels=zone_order, patch_artist=True,
                      notch=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2.5))
    for patch, z in zip(bp["boxes"], zone_order):
        patch.set_facecolor(ZCOL[z]); patch.set_alpha(0.75)
    ax.set_xlabel("Zone"); ax.set_ylabel("UCS (MPa)")
    ax.set_title("(b) Per-zone (boxplot)", fontsize=PANEL_SIZE, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax = fig.add_subplot(gs[2])
    for w in wells:
        m = df["Well"] == w
        ax.scatter(df.loc[m, "UCS_FINAL"], df.loc[m, "DEPTH_NORM"], s=12,
                   c=WCOL.get(w, "gray"), alpha=0.50, label=w)
    ax.set_xlabel("UCS (MPa)"); ax.set_ylabel("Normalized depth")
    ax.set_title("(c) UCS vs depth", fontsize=PANEL_SIZE, fontweight="bold")
    ax.invert_yaxis(); ax.set_xlim(0, 200)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[3])
    for z in zone_order:
        d = df[df["ZONE"] == z][["E_FINAL", "UCS_FINAL"]].dropna()
        ax.scatter(d["E_FINAL"], d["UCS_FINAL"], s=10, c=ZCOL[z], alpha=0.50, label=z)
    g = fit_info["NEW_GlobalStat_PL"]
    xr = np.linspace(0.5, 80, 300)
    ax.plot(xr, power_law(xr, g["a"], g["b"]), "k-", lw=3.0, label="Global PL")
    ax.set_xlabel("E_static (GPa)"); ax.set_ylabel("UCS (MPa)")
    ax.set_title("(d) UCS vs E_static", fontsize=PANEL_SIZE, fontweight="bold")
    ax.legend(loc="upper left"); ax.set_xlim(0, 80); ax.set_ylim(0, 200)

    plt.suptitle(f"Fig. 2  UCS data overview — distributions, depth trend, modulus relation\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", y=1.04)
    plt.tight_layout()
    return _save(fig, outdir, "Fig02_UCS_overview.png", dpi)

def fig03_obs_pred(df, res, zone_r2, outdir, dpi):
    published = res[res["family"] != "new"].head(6)["Key"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f"Fig. 3  Observed vs. predicted UCS — six highest-ranked published methods\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", y=1.02)
    zone_order = [z for z in ZONE_ORDER if (df["ZONE"] == z).sum() >= 10]

    for ai, key in enumerate(published):
        ax = axes[ai // 3, ai % 3]
        col = res.loc[res["Key"] == key, "column"].values[0]
        info = res[res["Key"] == key].iloc[0]
        valid = df[["UCS_FINAL", col, "ZONE"]].dropna()
        valid = valid[(valid["UCS_FINAL"] > 0) & (valid["UCS_FINAL"] < 300) &
                      (valid[col] > 0) & (valid[col] < 500)]
        for z in zone_order:
            zd = valid[valid["ZONE"] == z]
            ax.scatter(zd["UCS_FINAL"], zd[col], s=15, c=ZCOL[z], alpha=0.55, label=z)
        ax.plot([0, 200], [0, 200], "r-", lw=2.5, label="1:1")
        ymax = max(200.0, float(valid[col].quantile(0.995)) * 1.1)
        ax.text(0.04, 0.96,
                f"R² = {info['R2']:+.3f}\nRMSE = {info['RMSE']:.1f} MPa",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.9))
        ax.set_xlabel("UCS_FINAL (MPa)"); ax.set_ylabel("Predicted (MPa)")
        ax.set_title(f"({chr(97+ai)}) {info['Method'][:35]}",
                     fontsize=PANEL_SIZE, fontweight="bold")
        ax.set_xlim(0, 200); ax.set_ylim(0, ymax)
        if ai == 0:
            ax.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, "Fig03_obs_vs_pred_published.png", dpi)

def fig04_ranking(df, res, outdir, dpi):
    rd = res.copy()
    bar_colors = rd["color"].tolist()
    y_pos = np.arange(len(rd))
    labels = [f"#{int(r['Rank'])}  {r['Method'][:30]}" for _, r in rd.iterrows()]

    fig, axes = plt.subplots(1, 4, figsize=(26, 12))
    fig.suptitle(f"Fig. 4  Comprehensive ranking — R², RMSE, |Bias|, CCC\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", y=1.02)

    panels = [
        (axes[0], "(a) R² (strict)", "R²",         rd["R2"].values,          True),
        (axes[1], "(b) RMSE",        "RMSE (MPa)", rd["RMSE"].values,        False),
        (axes[2], "(c) |Bias|",      "|Bias| (MPa)", rd["Bias"].abs().values, False),
        (axes[3], "(d) CCC",         "CCC",        rd["CCC"].values,         False),
    ]
    for ax, title, xlabel, vals, show_yl in panels:
        vals = np.asarray(vals, float)
        bars = ax.barh(y_pos, vals, color=bar_colors, alpha=0.9,
                       edgecolor="black", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if show_yl else [""] * len(rd))
        ax.set_title(title, fontsize=PANEL_SIZE, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()

        vmin = float(np.nanmin(vals)); vmax = float(np.nanmax(vals))
        span = max(vmax - vmin, 1e-6)
        # Axis must span negative values; never clip at zero.
        lo = min(0.0, vmin - 0.08 * span)
        hi = vmax + 0.22 * span
        ax.set_xlim(lo, hi)

        if xlabel == "R²":
            ax.axvline(0, color="red", ls="--", lw=2, alpha=0.7)
        fmt = ".3f" if span <= 2 else ".1f"
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            off = 0.02 * span
            if val >= 0:
                ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                        f"{val:{fmt}}", va="center", ha="left",
                        fontsize=10, fontweight="bold")
            else:
                ax.text(val - off, bar.get_y() + bar.get_height() / 2,
                        f"{val:{fmt}}", va="center", ha="right",
                        fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, outdir, "Fig04_ranking.png", dpi)

def fig05_new_corr(df, res, fit_info, outdir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Fig. 5  New calibrated power-law UCS correlations — global and zone-specific\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", y=1.02)
    zone_order = [z for z in ZONE_ORDER if (df["ZONE"] == z).sum() >= 10]
    xr_es = np.linspace(0.5, 80, 400)
    xr_ed = np.linspace(0.5, 100, 400)

    ax = axes[0]
    for z in zone_order:
        d = df[df["ZONE"] == z].dropna(subset=["E_FINAL", "UCS_FINAL"])
        ax.scatter(d["E_FINAL"], d["UCS_FINAL"], s=10, c=ZCOL[z], alpha=0.45)
    g = fit_info["NEW_GlobalStat_PL"]
    r2g = res.loc[res["Key"] == "NEW_GlobalStat_PL", "R2"].values[0]
    ax.plot(xr_es, power_law(xr_es, g["a"], g["b"]), "k-", lw=3.5,
            label=f"Global Static:\n{g['eq']}\nR²={r2g:.3f}")
    ax.set_xlabel("E_static (GPa)"); ax.set_ylabel("UCS (MPa)")
    ax.set_title("(a) Global static-E PL", fontsize=PANEL_SIZE, fontweight="bold")
    ax.legend(loc="upper left"); ax.set_xlim(0, 80); ax.set_ylim(0, 200)

    ax = axes[1]
    for z in zone_order:
        d = df[df["ZONE"] == z].dropna(subset=["E_FINAL", "UCS_FINAL"])
        ax.scatter(d["E_FINAL"], d["UCS_FINAL"], s=10, c=ZCOL[z], alpha=0.40)
        ze = fit_info["NEW_ZoneSpec"]["zones"].get(z)
        if ze:
            r2z = res.loc[res["Key"] == "NEW_ZoneSpec", "zone_R2"].values[0].get(z, np.nan)
            ax.plot(xr_es, power_law(xr_es, ze["a"], ze["b"]), color=ZCOL[z], lw=3.5,
                    label=f"{z}: b={ze['b']:.2f} (R²={r2z:.2f})")
    ax.set_xlabel("E_static (GPa)"); ax.set_ylabel("UCS (MPa)")
    ax.set_title("(b) Zone-specific PL", fontsize=PANEL_SIZE, fontweight="bold")
    ax.legend(loc="upper left"); ax.set_xlim(0, 80); ax.set_ylim(0, 200)

    ax = axes[2]
    wells = sorted(df["Well"].unique())
    for w in wells:
        d = df[df["Well"] == w].dropna(subset=["YME_DYN", "UCS_FINAL"])
        ax.scatter(d["YME_DYN"], d["UCS_FINAL"], s=8, c=WCOL.get(w, "gray"),
                   alpha=0.35, label=w)
    gd = fit_info["NEW_GlobalDyn_PL"]
    r2d = res.loc[res["Key"] == "NEW_GlobalDyn_PL", "R2"].values[0]
    ax.plot(xr_ed, power_law(xr_ed, gd["a"], gd["b"]), "k-", lw=3.5,
            label=f"This study:\n{gd['eq']}\nR²={r2d:.3f}")
    lit = {
        "Militzer_1973": (2.922, 0.960, "Militzer & Stoll (1973)"),
        "Golubev_1976":  (0.0736, 1.673, "Golubev & Rabinovich (1976)"),
        "Chang_2006":    (13.8, 0.51, "Chang et al. (2006) carbonate"),
        "Lacy_1997":     (0.278, 1.549, "Lacy (1997)"),
        "Bradford_1998": (7.22, 0.712, "Bradford et al. (1998)"),
    }
    for key, (a, b, lbl) in lit.items():
        if key in res["Key"].values:
            ax.plot(xr_ed, np.clip(power_law(xr_ed, a, b), 0, 300), lw=2.0,
                    ls="--", alpha=0.85, color=METHOD_META[key]["color"], label=lbl[:25])
    ax.set_xlabel("E_dynamic (GPa)"); ax.set_ylabel("UCS (MPa)")
    ax.set_title("(c) Global dynamic vs literature", fontsize=PANEL_SIZE, fontweight="bold")
    ax.legend(loc="upper left"); ax.set_xlim(0, 100); ax.set_ylim(0, 250)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, "Fig05_new_correlations.png", dpi)

def fig06_heatmap(df, res, zone_r2, outdir, dpi):
    top = res.head(10)["Key"].tolist()
    zone_order = [z for z in ZONE_ORDER if (df["ZONE"] == z).sum() >= 10]
    hm = np.full((len(top), len(zone_order)), np.nan)
    for i, k in enumerate(top):
        for j, z in enumerate(zone_order):
            hm[i, j] = zone_r2.get(k, {}).get(z, np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))
    data_min = float(np.nanmin(hm)) if np.isfinite(hm).any() else -1.0
    vmin = max(-10.0, min(-2.0, data_min * 1.05))
    vmax = 1.0
    im = ax.imshow(hm, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(zone_order))); ax.set_xticklabels(zone_order)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([res.loc[res["Key"] == k, "Method"].values[0][:35] for k in top])

    for i in range(len(top)):
        for j in range(len(zone_order)):
            v = hm[i, j]
            if np.isnan(v):
                continue
            clipped = "*" if v < vmin else ""
            txt_col = "white" if v < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{v:+.2f}{clipped}", ha="center", va="center",
                    fontsize=10, color=txt_col, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("R² (strict)   [* = below color scale floor]")
    ax.set_title(f"Fig. 6  Zone-disaggregated R² heatmap — methods × zones\n{FIELD_NAME}",
                 fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, outdir, "Fig06_zone_heatmap.png", dpi)

def fig07_uncertainty(df, res, boot, outdir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Fig. 7  Uncertainty analysis — bootstrap stability and method spread\n{FIELD_NAME}",
                 fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    bs = boot.get("_boot_r2_array", None)
    if bs is not None and len(bs):
        ax.hist(bs, bins=40, color="#3498db", edgecolor="black", alpha=0.80)
        st = boot["global_static"]
        ax.axvline(st["mean"], color="red", ls="--", lw=2.5,
                   label=f"Mean = {st['mean']:.4f}")
        ax.axvline(st["ci95_lower"], color="orange", ls=":", lw=2,
                   label=f"95% CI [{st['ci95_lower']:.4f}, {st['ci95_upper']:.4f}]")
        ax.axvline(st["ci95_upper"], color="orange", ls=":", lw=2)
        ax.legend(loc="upper left")
    ax.set_xlabel("Bootstrap R² (global static PL)")
    ax.set_ylabel("Frequency")
    ax.set_title("(a) Bootstrap R² distribution", fontsize=PANEL_SIZE, fontweight="bold")

    ax = axes[1]
    mvals, lbls, clrs = {}, [], []
    for _, r in res.head(8).iterrows():
        v = df[r["column"]].dropna()
        v = v[(v > 0) & (v < 400)]
        if len(v) > 10:
            lbls.append(r["Method"][:25])
            mvals[r["Method"][:25]] = v.values
            clrs.append(r["color"])
    if mvals:
        data = [mvals[k] for k in lbls]
        vp = ax.violinplot(data, positions=range(len(lbls)),
                           showmedians=True, showextrema=True)
        for body, c in zip(vp["bodies"], clrs):
            body.set_facecolor(c); body.set_alpha(0.70)
            body.set_edgecolor("black"); body.set_linewidth(1.2)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=40, ha="right")
        ax.set_ylabel("Predicted UCS (MPa)")
    ax.set_title("(b) Method spread (top 8)", fontsize=PANEL_SIZE, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, "Fig07_uncertainty.png", dpi)

# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────
def write_paper_report(run_dir, df, res, fit_info, boot, cvlowo, zst,
                       t7_rows, csv_path, fnames, args):
    L = []
    A = L.append
    bar = "═" * 78
    A(bar); A(" UCS METHODS EVALUATION — RUN REPORT")
    A(f" generated: {datetime.now():%Y-%m-%d %H:%M:%S}"); A(bar)

    A("\n[0] RUN META"); A("-" * 78)
    A(f"   input CSV : {csv_path}")
    A(f"   rows      : {len(df):,}")
    A(f"   style/dpi : {args.style} / {args.dpi}")
    A(f"   wells     : {sorted(df['Well'].unique().tolist())}")

    A("\n[1] DATASET STRUCTURE"); A("-" * 78)
    for w in sorted(df['Well'].unique()):
        d = df[df['Well'] == w]
        A(f"   {w:10s}  n={len(d):>7,d}  depth {d['DEPTH'].min():.1f}-{d['DEPTH'].max():.1f} m")
    for z in ZONE_ORDER:
        d = df[df['ZONE'] == z]
        if len(d):
            A(f"   {z:10s}  n={len(d):>7,d}")

    A("\n[2] NEW CALIBRATIONS"); A("-" * 78)
    for k in ["NEW_GlobalStat_PL", "NEW_GlobalDyn_PL"]:
        fi = fit_info.get(k, {})
        A(f"   {k:20s} {fi.get('eq','')}   n={fi.get('n', '?')}")
    A("   NEW_ZoneSpec:")
    for r in t7_rows:
        A(f"   {r['Zone']:10s} {r['Equation']:32s} "
          f"b_CI95=[{r['b_CI95_low']:.3f}, {r['b_CI95_high']:.3f}] "
          f"zoneR²={r['zone_R2']:.4f} n={r['n']:,}")

    A("\n[3] METHOD RANKING"); A("-" * 78)
    A(f"   {'#':>2s} {'Key':18s} {'R2':>10s} {'RMSE':>8s} {'MAE':>8s} "
      f"{'Bias':>8s} {'r':>8s} {'CCC':>7s} {'n':>7s}")
    for _, r in res.iterrows():
        A(f"   {int(r['Rank']):>2d} {r['Key']:18s} {r['R2']:>10.4f} {r['RMSE']:>8.3f} "
          f"{r['MAE']:>8.3f} {r['Bias']:>8.3f} {r['Pearson_r']:>8.4f} {r['CCC']:>7.4f} {int(r['n']):>7,d}")

    A("\n[4] HONEST CV / LOWO VALIDATION"); A("-" * 78)
    A(f"   {'Method':20s} {'CV5_R2':>8s} {'CV5_RMSE':>9s} {'LOWO_R2':>8s} {'LOWO_RMSE':>10s}")
    for k, v in cvlowo.items():
        A(f"   {k:20s} {v.get('CV_R2', float('nan')):>8.4f} {v.get('CV_RMSE', float('nan')):>9.3f} "
          f"{v.get('LOWO_R2', float('nan')):>8.4f} {v.get('LOWO_RMSE', float('nan')):>10.3f}")

    A("\n[5] BOOTSTRAP UNCERTAINTY"); A("-" * 78)
    st = boot.get("global_static", {})
    A(f"   global-static R² : mean={st.get('mean', float('nan')):.4f}  "
      f"CI95=[{st.get('ci95_lower', float('nan')):.4f}, {st.get('ci95_upper', float('nan')):.4f}]  "
      f"n_boot={st.get('n_iter', '?')}")
    for z, zb in boot.get("zone_exponents", {}).items():
        A(f"   {z:10s} b = {zb.get('b_mean', float('nan')):.3f}  "
          f"CI95=[{zb.get('b_ci95', [float('nan')]*2)[0]:.3f}, "
          f"{zb.get('b_ci95', [float('nan')]*2)[1]:.3f}]")

    A("\n[6] FIGURE FILE MAP"); A("-" * 78)
    fmap = {"Fig01_workflow.png": "Fig. 1 — analytical workflow",
            "Fig02_UCS_overview.png": "Fig. 2 — UCS data overview",
            "Fig03_obs_vs_pred_published.png": "Fig. 3 — observed vs predicted (published)",
            "Fig04_ranking.png": "Fig. 4 — comprehensive ranking",
            "Fig05_new_correlations.png": "Fig. 5 — new calibrated correlations",
            "Fig06_zone_heatmap.png": "Fig. 6 — zone R² heatmap",
            "Fig07_uncertainty.png": "Fig. 7 — uncertainty analysis"}
    for f in fnames:
        A(f"   {f:36s} → {fmap.get(f, '')}")

    A(f"\n{bar}\n END OF REPORT\n{bar}")
    fp = run_dir / "PAPER_REPORT.txt"
    fp.write_text("\n".join(L), encoding="utf-8")
    log("    ✓ PAPER_REPORT.txt")
    return fp

# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global N_BOOT
    ap = argparse.ArgumentParser(description="UCS methods evaluation — figure generator")
    ap.add_argument("csv", nargs="?", default=None, help="path to master_dataset.csv")
    ap.add_argument("--base-dir", default=None, help="project root")
    ap.add_argument("--outdir", default=None, help="output directory")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", choices=["publication", "draft"], default="publication")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap iterations")
    args = ap.parse_args()
    N_BOOT = args.boot

    script_dir = Path(__file__).resolve().parent
    base_dir   = Path(args.base_dir) if args.base_dir else script_dir
    data_dir   = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv)
    else:
        cands = sorted(base_dir.rglob("master_dataset.csv"), key=lambda p: p.stat().st_mtime)
        csv_path = cands[-1] if cands else data_dir / "master_dataset.csv"

    # Fail-safe: if the CSV is missing, generate a synthetic placeholder for testing.
    if not csv_path.exists():
        log(f"  [INFO] master_dataset.csv not found at {csv_path}")
        log(f"  [INFO] Generating synthetic placeholder — DO NOT use for scientific reporting.")
        generate_synthetic_master(csv_path)

    run_name = f"run_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(args.outdir) if args.outdir else base_dir / "outputs" / run_name
    figs_dir   = run_dir / "figures"
    tables_dir = run_dir / "tables"
    json_dir   = run_dir / "json"
    logs_dir   = run_dir / "logs"
    for _d in (figs_dir, tables_dir, json_dir, logs_dir):
        _d.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PUB_RC)

    log("═" * 78)
    log(f"  UCS Methods Evaluation   |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"  input : {csv_path}")
    log(f"  output: {run_dir}")
    log("═" * 78)

    df = load_master(csv_path)
    log(f"\n[1] loaded: {len(df):,} rows | wells: {sorted(df['Well'].unique())}")
    colmap = resolve_columns(df)
    found = [k for k in METHOD_META if k in colmap]
    log(f"    method columns resolved: {len(found)}")

    log("\n[2] fitting new calibrations ...")
    fit_info = build_new_methods(df, colmap)
    for k in ["NEW_GlobalStat_PL", "NEW_GlobalDyn_PL"]:
        log(f"    {fit_info[k]['eq']}")
    for z, ze in fit_info["NEW_ZoneSpec"]["zones"].items():
        log(f"    {z:9s}: {ze['eq']}   (n={ze['n']})")

    log("\n[3] evaluating all methods ...")
    res, zone_r2 = evaluate_all(df, colmap)
    log(res[["Rank", "Key", "R2", "RMSE", "MAE", "Bias", "CCC"]].to_string(index=False))

    log("\n[4] CV / LOWO for fitted methods ...")
    cvlowo = cv_lowo_fitted(df, fit_info)
    for k, v in cvlowo.items():
        log(f"    {k:18s}: {v}")

    log(f"\n[5] bootstrap ({args.boot} iter) ...")
    boot = bootstrap_block(df, fit_info, n_boot=args.boot)
    log(f"    global-static R² CI95=[{boot['global_static']['ci95_lower']:.4f}, "
        f"{boot['global_static']['ci95_upper']:.4f}]")

    zst = zone_stats_table(df)

    log("\n[6] figures ...")
    fnames = []
    fnames.append(fig01_workflow(df, res, fit_info, figs_dir, args.dpi))
    fnames.append(fig02_overview(df, res, fit_info, figs_dir, args.dpi))
    fnames.append(fig03_obs_pred(df, res, zone_r2, figs_dir, args.dpi))
    fnames.append(fig04_ranking(df, res, figs_dir, args.dpi))
    fnames.append(fig05_new_corr(df, res, fit_info, figs_dir, args.dpi))
    fnames.append(fig06_heatmap(df, res, zone_r2, figs_dir, args.dpi))
    fnames.append(fig07_uncertainty(df, res, boot, figs_dir, args.dpi))

    log("\n[7] tables ...")
    t5 = res[["Rank", "Key", "Method", "family", "R2", "RMSE", "MAE", "Bias",
              "Pearson_r", "CCC", "n"]].copy()
    t5.to_csv(tables_dir / "Table5_method_metrics.csv", index=False)

    t7_rows = []
    for z, ze in fit_info["NEW_ZoneSpec"]["zones"].items():
        ci = boot["zone_exponents"].get(z, {}).get("b_ci95", [np.nan, np.nan])
        zr = zone_r2.get("NEW_ZoneSpec", {}).get(z, np.nan)
        t7_rows.append({"Zone": z, "Equation": ze["eq"], "a": ze["a"], "b": ze["b"],
                        "b_CI95_low": ci[0], "b_CI95_high": ci[1],
                        "zone_R2": zr, "n": ze["n"]})
    pd.DataFrame(t7_rows).to_csv(tables_dir / "Table7_zone_calibrations.csv", index=False)
    zst.to_csv(tables_dir / "Table9_zone_stats.csv", index=False)

    def clean(o):
        if isinstance(o, dict):  return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):  return [clean(x) for x in o]
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o

    summary = {
        "meta": {"generator": "ucs_methods_evaluation.py",
                 "run_ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
                 "field": FIELD_NAME, "input_csv": str(csv_path),
                 "style": args.style, "dpi": args.dpi},
        "dataset": {"n_total": int(len(df)),
                    "wells": sorted(df["Well"].unique().tolist()),
                    "per_zone_n": {z: int((df["ZONE"] == z).sum()) for z in ZONE_ORDER}},
        "new_equations": clean(fit_info),
        "ranking": clean(res.drop(columns=["zone_R2"]).to_dict(orient="records")),
        "zone_R2": clean(zone_r2),
        "cv_lowo": clean(cvlowo),
        "bootstrap": clean(boot),
        "zone_stats": clean(zst.to_dict(orient="records")),
        "figures": fnames,
    }
    with open(json_dir / "figures_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_paper_report(run_dir, df, res, fit_info, boot, cvlowo, zst,
                       t7_rows, csv_path, fnames, args)
    (logs_dir / "run_log.txt").write_text("\n".join(LOG_BUFFER), encoding="utf-8")

    log("\n" + "═" * 78)
    log(f"  COMPLETE — {len(fnames)} figures + 3 tables + summary JSON")
    log(f"  run folder: {run_dir}")
    log("═" * 78)


if __name__ == "__main__":
    main()
