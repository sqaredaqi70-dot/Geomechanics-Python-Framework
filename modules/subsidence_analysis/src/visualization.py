"""
Subsidence Visualization Module
================================
Publication-quality plotting utilities for subsidence analysis results.
Supports 1D profiles, 2D subsidence maps, 3D deformed surfaces, and 
statistical distributions from Monte Carlo simulations.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import gaussian_kde
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
CM_TO_INCH = 0.393701
MPA_TO_PSI = 145.0377

# Default publication-quality style
PUB_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Standard risk classification (subsidence thresholds in cm)
RISK_THRESHOLDS_CM = [0, 1, 3, 5, 10]
RISK_LABELS = ["Negligible", "Low", "Medium", "High", "Very High"]
RISK_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8b0000"]

# Custom colormap for subsidence
SUBSIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "subsidence",
    ["#ffffff", "#a8d5f0", "#3498db", "#27ae60",
     "#f1c40f", "#f39c12", "#e74c3c", "#8b0000"]
)


def apply_publication_style():
    """Apply the default publication-quality Matplotlib style."""
    plt.rcParams.update(PUB_STYLE)


# ══════════════════════════════════════════════════════════════
# 1D & PROFILE PLOTS
# ══════════════════════════════════════════════════════════════
def plot_radial_profile(
    radial_distance_km: np.ndarray,
    subsidence_cm: np.ndarray,
    title: str = "Radial Subsidence Profile",
    label: str = "Subsidence",
    color: str = "#0072B2",
    show_inch_axis: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a 1D radial subsidence profile with optional secondary axis in inches.

    Parameters
    ----------
    radial_distance_km : np.ndarray
        Radial distance from the reservoir center (km).
    subsidence_cm : np.ndarray
        Subsidence values corresponding to each distance (cm).
    title : str
        Plot title.
    label : str
        Legend label for the curve.
    color : str
        Line color (hex or named color).
    show_inch_axis : bool
        If True, adds a secondary Y-axis in inches.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(radial_distance_km, subsidence_cm, color=color, lw=2.5, label=label)
    ax.fill_between(radial_distance_km, 0, subsidence_cm, alpha=0.15, color=color)

    ax.set_xlabel("Radial distance from center (km)")
    ax.set_ylabel("Subsidence (cm)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")

    if show_inch_axis:
        ax_r = ax.twinx()
        ax_r.set_ylim([v * CM_TO_INCH for v in ax.get_ylim()])
        ax_r.set_ylabel("Subsidence (inch)")

    return fig


def plot_subsidence_vs_depletion(
    depletion_mpa: np.ndarray,
    subsidence_cm: np.ndarray,
    title: str = "Subsidence vs. Pressure Depletion",
    label: str = "Model output",
    color: str = "#D55E00",
    show_psi_axis: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot subsidence as a function of reservoir pressure depletion.

    Parameters
    ----------
    depletion_mpa : np.ndarray
        Reservoir depletion values (MPa).
    subsidence_cm : np.ndarray
        Corresponding subsidence (cm).
    title, label, color : str
        Plot labels.
    show_psi_axis : bool
        Adds a secondary X-axis in psi.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(depletion_mpa, subsidence_cm, "o-", color=color, lw=2.5, ms=8, label=label)
    ax.set_xlabel("Depletion ΔPp (MPa)")
    ax.set_ylabel("Subsidence (cm)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=":")

    if show_psi_axis:
        ax_top = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x * MPA_TO_PSI, lambda x: x / MPA_TO_PSI)
        )
        ax_top.set_xlabel("ΔPp (psi)")

    return fig


# ══════════════════════════════════════════════════════════════
# 2D SUBSIDENCE MAPS
# ══════════════════════════════════════════════════════════════
def plot_subsidence_map(
    x_km: np.ndarray,
    y_km: np.ndarray,
    subsidence_cm: np.ndarray,
    title: str = "Surface Subsidence Map",
    well_coords_km: Optional[Dict[str, Tuple[float, float]]] = None,
    cmap: str = "YlOrRd",
    show_contours: bool = True,
    n_contours: int = 5
) -> plt.Figure:
    """
    Plot a 2D subsidence map with contours and optional well markers.

    Parameters
    ----------
    x_km, y_km : np.ndarray
        2D grid arrays of Easting/Northing coordinates in km.
    subsidence_cm : np.ndarray
        2D subsidence values in cm (NaN outside reservoir).
    title : str
        Plot title.
    well_coords_km : dict, optional
        Dictionary of well names and (x, y) locations in km.
    cmap : str
        Matplotlib colormap name.
    show_contours : bool
        Overlay contour lines on the map.
    n_contours : int
        Number of contour levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    v_max = float(np.nanmax(subsidence_cm)) if np.any(np.isfinite(subsidence_cm)) else 1.0

    im = ax.pcolormesh(x_km, y_km, subsidence_cm, cmap=cmap, shading="auto",
                       vmin=0, vmax=v_max)

    if show_contours and v_max > 0:
        levels = np.linspace(v_max * 0.2, v_max * 0.9, n_contours)
        cs = ax.contour(x_km, y_km, subsidence_cm, levels=levels,
                        colors="white", linewidths=1.2)
        ax.clabel(cs, fmt="%.2f", fontsize=9)

    if well_coords_km:
        markers = ["o", "s", "^", "D", "P", "*"]
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        for i, (name, (xw, yw)) in enumerate(well_coords_km.items()):
            m = markers[i % len(markers)]
            c = colors[i % len(colors)]
            ax.scatter(xw, yw, s=200, c=c, marker=m,
                       edgecolors="black", linewidths=2, zorder=10)
            ax.annotate(name, (xw, yw), fontsize=11, fontweight="bold",
                        xytext=(8, 8), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=c, alpha=0.9))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Subsidence (cm)", fontsize=12)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    return fig


def plot_risk_zonation_map(
    x_km: np.ndarray,
    y_km: np.ndarray,
    subsidence_cm: np.ndarray,
    title: str = "Subsidence Risk Zonation",
    well_coords_km: Optional[Dict[str, Tuple[float, float]]] = None
) -> plt.Figure:
    """
    Plot a discrete risk classification map (Negligible/Low/Medium/High/Very High).

    Parameters
    ----------
    x_km, y_km, subsidence_cm : np.ndarray
        Grid arrays.
    title : str
        Plot title.
    well_coords_km : dict, optional
        Well positions.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    risk = np.digitize(subsidence_cm, RISK_THRESHOLDS_CM) - 1
    risk = np.clip(risk, 0, len(RISK_LABELS) - 1).astype(float)
    risk[~np.isfinite(subsidence_cm)] = np.nan

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(RISK_COLORS)
    im = ax.pcolormesh(x_km, y_km, risk, cmap=cmap, shading="auto",
                       vmin=0, vmax=len(RISK_LABELS) - 1)

    if well_coords_km:
        for name, (xw, yw) in well_coords_km.items():
            ax.scatter(xw, yw, s=200, c="black", marker="o",
                       edgecolors="white", linewidths=2, zorder=10)
            ax.annotate(name, (xw, yw), fontsize=11, fontweight="bold",
                        xytext=(8, 8), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(RISK_COLORS, RISK_LABELS)]
    ax.legend(handles=legend_patches, loc="upper right", title="Risk Class")

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    return fig


# ══════════════════════════════════════════════════════════════
# 3D SURFACE PLOTS
# ══════════════════════════════════════════════════════════════
def plot_3d_deformed_surface(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_depth_m: np.ndarray,
    subsidence_cm: np.ndarray,
    title: str = "3D Deformed Reservoir Surface",
    cmap: str = "YlOrRd",
    elev: int = 25,
    azim: int = -55,
    ghost_original: bool = True
) -> plt.Figure:
    """
    Render a 3D reservoir surface deformed by the subsidence values, 
    with the original horizon shown as a ghost outline.

    Parameters
    ----------
    x_km, y_km : np.ndarray
        Coordinate grids in km.
    z_depth_m : np.ndarray
        Reservoir top depth in meters (positive down).
    subsidence_cm : np.ndarray
        Subsidence to apply to each cell (cm).
    title : str
        Plot title.
    cmap : str
        Colormap for subsidence.
    elev, azim : int
        3D viewpoint angles.
    ghost_original : bool
        Show the pre-subsidence surface as a semi-transparent ghost.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Downsample for performance
    step = max(1, x_km.shape[0] // 80)
    xs = x_km[::step, ::step]
    ys = y_km[::step, ::step]
    zs = z_depth_m[::step, ::step]
    sub_s = subsidence_cm[::step, ::step]

    v_max = float(np.nanmax(subsidence_cm)) if np.any(np.isfinite(subsidence_cm)) else 1.0
    norm = Normalize(vmin=0, vmax=max(0.001, v_max))
    cmo = cm.get_cmap(cmap)
    rgba = cmo(norm(sub_s))
    rgba[..., -1] = np.where(np.isfinite(sub_s), 0.88, 0.0)

    # Compute deformed surface (subsidence in meters, added to depth)
    z_deformed = (z_depth_m - subsidence_cm / 100.0)[::step, ::step]

    if ghost_original:
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.15,
                        linewidth=0, antialiased=False)

    ax.plot_surface(xs, ys, z_deformed, facecolors=rgba,
                    linewidth=0, antialiased=True, shade=True, alpha=0.92)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label("Subsidence (cm)", fontsize=12)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Easting (km)", labelpad=10)
    ax.set_ylabel("Northing (km)", labelpad=10)
    ax.set_zlabel("Depth (m)", labelpad=10)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.invert_zaxis()

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_alpha(0.03)

    return fig


# ══════════════════════════════════════════════════════════════
# MONTE CARLO STATISTICAL PLOTS
# ══════════════════════════════════════════════════════════════
def plot_mc_distributions(
    mc_results: Dict[float, Dict],
    title: str = "Monte Carlo Subsidence Distributions",
    show_risk_zones: bool = True
) -> plt.Figure:
    """
    Plot PDF distributions from Monte Carlo results for multiple scenarios,
    with mean lines and optional risk zone shading.

    Parameters
    ----------
    mc_results : dict
        Output from `MonteCarloSubsidence.run_all_scenarios()`.
    title : str
        Plot title.
    show_risk_zones : bool
        Overlay colored bands for risk thresholds.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    colors_seq = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#8e44ad", "#1abc9c"]

    for i, (dpp, stats) in enumerate(mc_results.items()):
        arr = stats["sub_samples_cm"]
        c = colors_seq[i % len(colors_seq)]

        ax.hist(arr, bins=60, density=True, alpha=0.20, color=c, edgecolor="none")
        kde = gaussian_kde(arr)
        x = np.linspace(arr.min(), arr.max(), 300)
        ax.plot(x, kde(x), color=c, lw=2.5,
                label=f"ΔPp={dpp} MPa | μ={stats['mean']:.2f}±{stats['std']:.2f} cm")
        ax.axvline(stats["mean"], color=c, ls="--", lw=1.2, alpha=0.7)

    if show_risk_zones:
        for lo, hi, fc in zip([0, 1, 3, 5, 10], [1, 3, 5, 10, 20], RISK_COLORS):
            ax.axvspan(lo, hi, alpha=0.07, color=fc)

    ax.set_xlabel("Surface Subsidence (cm)")
    ax.set_ylabel("Probability Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    return fig


def plot_tornado_sensitivity(
    tornado_df: pd.DataFrame,
    baseline: float,
    title: str = "Tornado Sensitivity Analysis"
) -> plt.Figure:
    """
    Plot a Tornado diagram from sensitivity analysis output.

    Parameters
    ----------
    tornado_df : pd.DataFrame
        Must contain columns: 'parameter', 'low_cm', 'high_cm', 'swing_cm'.
    baseline : float
        Base subsidence value (cm) at mean parameters.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    df = tornado_df.sort_values("swing_cm", ascending=True)
    y = np.arange(len(df))

    for i, row in enumerate(df.itertuples()):
        lo_width = baseline - row.low_cm
        hi_width = row.high_cm - baseline
        ax.barh(i, lo_width, left=row.low_cm, color="#3498db",
                edgecolor="black", alpha=0.85)
        ax.barh(i, hi_width, left=baseline, color="#e74c3c",
                edgecolor="black", alpha=0.85)
        ax.text(row.low_cm - 0.02, i, f"{row.low_cm:.3f}",
                va="center", ha="right", fontsize=10)
        ax.text(row.high_cm + 0.02, i, f"{row.high_cm:.3f}",
                va="center", ha="left", fontsize=10)

    ax.axvline(baseline, color="black", lw=2, ls="--",
               label=f"Base = {baseline:.3f} cm")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Subsidence (cm)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    return fig


def plot_sobol_indices(
    sobol_df: pd.DataFrame,
    title: str = "Sobol Sensitivity Indices"
) -> plt.Figure:
    """
    Plot Sobol first-order (S1) and total-order (ST) sensitivity indices.

    Parameters
    ----------
    sobol_df : pd.DataFrame
        Must contain columns: 'parameter', 'S1', 'ST'.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    df = sobol_df.sort_values("ST", ascending=True)
    y = np.arange(len(df))
    width = 0.35

    ax.barh(y - width/2, df["S1"], width, color="#3498db",
            edgecolor="black", label="S₁ (first-order)")
    ax.barh(y + width/2, df["ST"], width, color="#e74c3c",
            edgecolor="black", label="Sₜ (total-order)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Sobol Index")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    return fig


# ══════════════════════════════════════════════════════════════
# METHOD COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════
def plot_method_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Multi-Model Subsidence Comparison",
    baseline_col: str = "Ratio_to_Geertsma"
) -> plt.Figure:
    """
    Bar chart comparing multiple subsidence models with ratio subplot.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of `SubsidenceComparator.get_comparison_table()`.
    title : str
        Plot title.
    baseline_col : str
        Column name containing ratios to baseline.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    fig.suptitle(title, fontsize=15, fontweight="bold")

    df = comparison_df.sort_values("Max_Subsidence_cm", ascending=True)
    names = df["Method"]
    vals = df["Max_Subsidence_cm"]

    # (a) Absolute values
    ax = axes[0]
    colors_seq = plt.cm.tab20(np.linspace(0, 1, len(names)))
    bars = ax.barh(range(len(names)), vals, color=colors_seq,
                   edgecolor="black", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f} cm", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Max Subsidence (cm)")
    ax.set_title("(a) Absolute peak subsidence", fontweight="bold")

    # (b) Ratios
    ax = axes[1]
    if baseline_col in df.columns:
        ratios = df[baseline_col].fillna(0)
        ax.barh(range(len(names)), ratios, color=colors_seq,
                edgecolor="black", alpha=0.7)
        ax.axvline(1.0, color="black", lw=2, ls="--", label="Baseline = 1.0")
        for i, r in enumerate(ratios):
            if r > 0:
                ax.text(r + 0.02, i, f"{r:.3f}", va="center", fontsize=10)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(f"Ratio to baseline")
        ax.set_title("(b) Normalized comparison", fontweight="bold")
        ax.legend()

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════
def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Union[str, Path] = "./figures",
    dpi: int = 300,
    close_after: bool = True
) -> Path:
    """
    Save a figure with sanitized filename and specified DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Name for the output file (without extension).
    output_dir : str or Path
        Directory to save the figure into (created if missing).
    dpi : int
        Resolution.
    close_after : bool
        Close the figure after saving to free memory.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in filename)
    file_path = out_dir / f"{safe_name}.png"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    if close_after:
        plt.close(fig)
    return file_path
