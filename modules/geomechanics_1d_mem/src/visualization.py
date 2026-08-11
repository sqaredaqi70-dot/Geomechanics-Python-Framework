                    VISUALIZATION MODULE

Plotting utilities for the 1D Mechanical Earth Model workflow.

Author:  Saeed Gharedaghi
Contact: sqaredaqi70@gmail.com
License: AGPL-3.0 (Academic) / Commercial (see LICENSE files)
═══════════════════════════════════════════════════════════════════════
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from . import config as cfg


# ══════════════════════════════════════════════════════════════
# GLOBAL PLOT SETTINGS
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          20,
    'axes.labelsize':     22,
    'axes.titlesize':     24,
    'legend.fontsize':    18,
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     ':',
})


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def save_figure(fig, name, output_dir, dpi=300):
    """
    Save a figure with a sanitized filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Figure name (will be sanitized).
    output_dir : Path
        Output directory.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved figure.
    """
    safe = (name.replace(' ', '_').replace('/', '_')
            .replace('\\', '_').replace(':', '')
            .replace(',', '').replace('(', '')
            .replace(')', ''))[:140]
    path = Path(output_dir) / f"{safe}.png"
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {safe}.png ({path.stat().st_size // 1024} KB)")
    return path


def add_zone_backgrounds(ax, df, depth_col='DEPTH_ft', alpha=0.10):
    """Add zone-colored horizontal bands to an axis."""
    for zone in cfg.ZONE_ORDER:
        zd = df[df['ZONE'] == zone]
        if len(zd) < 2:
            continue
        ax.axhspan(
            zd[depth_col].min(), zd[depth_col].max(),
            alpha=alpha,
            color=cfg.ZONE_FACECOLORS.get(zone, 'white'),
            zorder=0,
        )


# 1D MEM PROFILE PLOT (per well)
def plot_mem_profile(well_data, output_dir):
    """
    Create a multi-track 1D MEM profile for one well.

    Parameters
    ----------
    well_data : dict
        Processed well data.
    output_dir : Path
        Directory to save the figure.
    """
    df = well_data['df']
    name = well_data['name']
    color = cfg.WELL_COLORS.get(name, '#333333')

    fig, axes = plt.subplots(1, 5, figsize=(28, 16), sharey=True)

    # Track 1: Dynamic moduli
    ax = axes[0]
    add_zone_backgrounds(ax, df)
    for col, clr, lbl in [
        ('E_dyn_Mpsi', 'steelblue', 'E$_{dyn}$'),
        ('G_dyn_Mpsi', '#ff7f0e',   'G$_{dyn}$'),
        ('K_dyn_Mpsi', '#2ca02c',   'K$_{dyn}$'),
    ]:
        if col in df.columns:
            ax.plot(df[col], df['DEPTH_ft'], color=clr,
                    lw=1.2, label=lbl)
    ax.set_xlabel('Modulus (Mpsi)')
    ax.set_ylabel('Depth (ft MD)')
    ax.set_title('Dynamic Moduli')
    ax.legend(fontsize=12)
    ax.invert_yaxis()

    # Track 2: UCS
    ax = axes[1]
    add_zone_backgrounds(ax, df)
    if 'UCS_MPa' in df.columns:
        ax.plot(df['UCS_MPa'].clip(0, 150), df['DEPTH_ft'],
                color=color, lw=1.5)
    ax.set_xlabel('UCS (MPa)')
    ax.set_title('UCS')
    ax.invert_yaxis()

    # Track 3: Stresses
    ax = axes[2]
    add_zone_backgrounds(ax, df)
    for col, sty, lbl in [
        ('Sv_psi', 'k-',  'S$_v$'),
        ('SH_psi', 'r-',  'σ$_H$'),
        ('Sh_psi', 'b-',  'σ$_h$'),
        ('Pp_psi', 'g--', 'P$_p$'),
    ]:
        if col in df.columns:
            ax.plot(df[col], df['DEPTH_ft'], sty, lw=1.2, label=lbl)
    ax.set_xlabel('Stress (psi)')
    ax.set_title('Principal Stresses')
    ax.legend(fontsize=12)
    ax.invert_yaxis()

    # Track 4: Stress regime
    ax = axes[3]
    add_zone_backgrounds(ax, df)
    if 'REGIME' in df.columns:
        regime_map = {'NF': 0, 'SS': 1, 'RF': 2}
        regime_clr = {'NF': 'steelblue', 'SS': '#2ca02c', 'RF': '#d62728'}
        for reg, xi in regime_map.items():
            mask = df['REGIME'] == reg
            if mask.any():
                ax.scatter(np.full(int(mask.sum()), xi),
                           df['DEPTH_ft'][mask],
                           c=regime_clr[reg], s=6, alpha=0.5, label=reg)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['NF', 'SS', 'RF'])
        ax.legend(fontsize=12, markerscale=3)
    ax.set_title('Stress Regime')
    ax.invert_yaxis()

    # Track 5: Mud weight window
    ax = axes[4]
    add_zone_backgrounds(ax, df)
    if 'MW_col_ppg' in df.columns:
        mc = df['MW_col_ppg'].replace([np.inf, -np.inf], np.nan)
        mf = df['MW_frc_ppg'].replace([np.inf, -np.inf], np.nan)
        mp = df['MW_pp_ppg'].replace([np.inf, -np.inf], np.nan)
        mmod = df['MW_moderated_collapse_ppg'].replace(
            [np.inf, -np.inf], np.nan
        )
        ok = (mc > 4) & (mc < 25) & (mf > 4) & (mf < 25)

        if ok.any():
            ax.fill_betweenx(df['DEPTH_ft'][ok], mc[ok], mf[ok],
                              alpha=0.2, color='#FFFACD',
                              label='Safe window')
            ax.plot(mp[ok], df['DEPTH_ft'][ok], 'g-', lw=1.2,
                    label='P$_p$ eq.')
            ax.plot(mc[ok], df['DEPTH_ft'][ok], 'r-', lw=1.5,
                    label='Collapse')
            ax.plot(mmod[ok], df['DEPTH_ft'][ok], color='#8B0000',
                    lw=2.0, ls='-.', label='Mod. collapse')
            ax.plot(mf[ok], df['DEPTH_ft'][ok], 'b-', lw=1.5,
                    label='Fracture')

    ax.set_xlabel('MW (ppg)')
    ax.set_title('Mud Weight Window')
    ax.legend(fontsize=10, loc='lower right')
    ax.invert_yaxis()

    fig.suptitle(f'1D MEM Profile — {name}',
                 fontsize=26, fontweight='bold',
                 color=color, y=0.995)
    plt.tight_layout()
    save_figure(fig, f"Fig_MEM_Profile_{name}", output_dir)


# STRESS POLYGON
def plot_stress_polygon(wells, regime_info, output_dir):
    """
    Plot stress polygon showing regime classification.

    Parameters
    ----------
    wells : dict
        Dictionary of well data.
    regime_info : dict
        Output from determine_overall_regime().
    output_dir : Path
        Output directory.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot data points from all wells
    for name, wd in wells.items():
        df = wd['df']
        if 'Sv_MPa' not in df.columns:
            continue
        sv = df['Sv_MPa'].replace(0, np.nan)
        if 'SHMIN_PHS' not in df.columns or 'SHMAX_PHS' not in df.columns:
            continue
        shr = df['SHMIN_PHS'] / sv
        sHr = df['SHMAX_PHS'] / sv
        m = (shr > 0) & (shr < 3) & (sHr > 0) & (sHr < 3)
        ax.scatter(shr[m][::5], sHr[m][::5], s=8,
                   c=cfg.WELL_COLORS.get(name, 'gray'),
                   alpha=0.35, label=name)

    # Regime regions
    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color='blue',   alpha=0.05)
    ax.fill([0, 1, 1, 0], [1, 1, 3, 3], color='orange', alpha=0.05)
    ax.fill([1, 3, 3, 1], [1, 1, 3, 3], color='red',    alpha=0.05)
    ax.axhline(1, color='navy',   lw=1.2, alpha=0.4)
    ax.axvline(1, color='maroon', lw=1.2, alpha=0.4)

    # Regime labels
    for txt, x_, y_, c_ in [
        ('NF', 0.4, 0.4, 'blue'),
        ('SS', 0.4, 1.8, 'orange'),
        ('RF', 1.8, 2.2, 'red'),
    ]:
        ax.text(x_, y_, txt, fontsize=28, alpha=0.6,
                style='italic', color=c_)

    # Overall regime marker
    K0 = regime_info['K0']
    kH = regime_info['kH']
    if np.isfinite(K0) and np.isfinite(kH):
        ax.plot(K0, kH, 'k*', ms=26, zorder=10)
        ax.annotate(
            (f"[{regime_info['regime_code']}]\n"
             f"K₀={K0:.2f}\nkH={kH:.2f}"),
            xy=(K0, kH),
            xytext=(K0 + 0.4, kH - 0.4),
            fontsize=18, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2.5),
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9)
        )

    ax.set_xlabel('K₀ = σ$_h$/S$_v$')
    ax.set_ylabel('σ$_H$/S$_v$')
    ax.set_title(f"Stress Polygon\n{regime_info['regime_str']}",
                 fontweight='bold')
    ax.legend(fontsize=16)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    save_figure(fig, "Fig_Stress_Polygon", output_dir)


# WELL LOCATION MAP
def plot_well_locations(wells, output_dir):
    """Plot map of well locations."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for name, wd in wells.items():
        x, y = wd['position']
        color = cfg.WELL_COLORS.get(name, '#333333')
        marker = cfg.WELL_MARKERS.get(name, 'o')
        ax.scatter(x, y, s=500, c=color, marker=marker,
                   edgecolors='black', linewidths=2.5, zorder=10)
        ax.annotate(name, (x, y), fontsize=18, fontweight='bold',
                    color=color, xytext=(14, 14),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              edgecolor=color, alpha=0.9))

    ax.set_xlabel('Easting (km, relative)')
    ax.set_ylabel('Northing (km, relative)')
    ax.set_title('Well Locations (Relative Positions)',
                 fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    save_figure(fig, "Fig_Well_Locations", output_dir)
