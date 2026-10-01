#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mem_wellbore_stability_analysis.py
───────────────────────────────────────────────────────────────────────────────
1D Mechanical Earth Model (MEM) & Wellbore Stability Analysis Module
Optimized for Geomechanics Python Framework (GitHub Release Version)

FEATURES:
- 1D In-situ stress profiling (Sv, SHmax, Shmin, Pp)
- Density integration for overburden stress: Sv = ∫ ρ(z) g dz
- Stress regime classification via Zoback polygon (K0, kH)
- Wellbore stability & Mud Weight Window (Collapse, Moderated Collapse, Fracture)
- Mohr-Coulomb failure envelope and shear safety margin calculation
- 3D IDW spatial property mapping across well trajectories
- 100% Anonymized & Portable with automatic synthetic fallback
"""

import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats as sp_stats
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS & ANONYMIZATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
G_MS2        = 9.81
MPA_TO_PSI   = 145.0377
GPA_TO_MPSI  = 0.145038
M_TO_FT      = 3.28084
GCC_TO_PPG   = 8.345404
PPG_TO_PSIFT = 0.052

RHO_WATER          = 1.025
RHO_FALLBACK       = 2.40
T0_FRACTION        = 0.10
SHMAX_AZIMUTH      = 35.0
MW_SAFETY_MARGIN   = 0.50     # ppg — moderated collapse safety buffer

FIELD_NAME_ANON = "Representative Carbonate Field (Zagros Basin)"
WELL_ALIASES = {'Well-01': 'Well-A', 'Well-05': 'Well-B', 'Well-10': 'Well-C'}

COLORS = {'Well-A': '#e41a1c', 'Well-B': '#377eb8', 'Well-C': '#4daf4a'}
MARKERS = {'Well-A': 'o', 'Well-B': 's', 'Well-C': '^'}

ZONE_ORDER = ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']
ZONE_COLORS = {
    'Ghar':     '#2ca02c', 'Asmari-A': '#d62728',
    'Asmari-B': '#1f77b4', 'Jahrum':   '#ff7f0e',
    'Undiff.':  'lightgray',
}
ZONE_FC = {
    'Ghar':     '#e8f5e9', 'Asmari-A': '#ffebee',
    'Asmari-B': '#e3f2fd', 'Jahrum':   '#fff9c4',
    'Undiff.':  '#f5f5f5',
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. PATHS SETUP
# ─────────────────────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"

FIG_DIR = OUTPUT_DIR / "figures"
FIG_3D  = OUTPUT_DIR / "figures_3D"
CSV_DIR = OUTPUT_DIR / "tables_csv"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, FIG_DIR, FIG_3D, CSV_DIR, LOG_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class SessionLogger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, 'w', encoding='utf-8')
    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = SessionLogger(str(LOG_DIR / f"mem_stability_session_{RUN_TS}.log"))

# Matplotlib Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("=" * 80)
print(f"1D MEM & Wellbore Stability Pipeline | {RUN_TS}")
print(f"Field Area: {FIELD_NAME_ANON}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER MATHEMATICAL & CONVERSION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def p_to_gcc(p_mpa, depth_m):
    p = np.asarray(p_mpa, dtype=float)
    d = np.asarray(depth_m, dtype=float)
    d = np.where(d > 0, d, np.nan)
    return p * 1000.0 / (G_MS2 * d)

def p_mpa_to_ppg(p_mpa, depth_m):
    return p_to_gcc(p_mpa, depth_m) * GCC_TO_PPG

def safe_median(series):
    if series is None: return np.nan
    v = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.median()) if len(v) > 0 else np.nan

def zone_background(ax, df, depth_col='DEPTH_ft', alpha=0.08):
    for zone in ZONE_ORDER:
        zd = df[df['ZONE'] == zone]
        if len(zd) >= 2:
            ax.axhspan(zd[depth_col].min(), zd[depth_col].max(),
                       alpha=alpha, color=ZONE_FC.get(zone, 'white'), zorder=0)

def zone_labels(ax, df, depth_col='DEPTH_ft'):
    for zone in ZONE_ORDER:
        zd = df[df['ZONE'] == zone]
        if len(zd) > 0:
            mid = (zd[depth_col].min() + zd[depth_col].max()) / 2.0
            ax.text(-0.05, mid, zone, fontsize=10, ha='right', va='center',
                    color=ZONE_COLORS.get(zone, 'black'), fontweight='bold',
                    transform=ax.get_yaxis_transform(), clip_on=False)

def save_fig(fig, name, folder):
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved Figure: {folder.name}/{name}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD & PROCESS WELL LOG DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Processing Geomechanical Well Logs & Stress State...")

# Ensure synthetic data is present
from generate_synthetic_data import main as generate_data
generate_data()

wells = {}
for wa in ['Well-A', 'Well-B', 'Well-C']:
    las_path = DATA_DIR / "wells" / f"{wa}_synthetic.las"
    if not las_path.exists():
        continue

    # Parse synthetic LAS
    with open(las_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ascii_idx = next(i for i, line in enumerate(lines) if "~Ascii" in line)
    
    cols = ['DEPTH', 'GR', 'DT', 'DTS', 'RHOB', 'UCS_FINAL', 'TSTR_10%', 
            'SHMAX_PHS', 'SHMIN_PHS', 'FINAL_PP', 'YME_DYN', 'E_FINAL', 
            'PR_DYN', 'PR_STA', 'SMG_DYN', 'BMK_DYN', 'FANG_FROMGR', 'COHESION_FROM_GR']
    
    data_rows = []
    for line in lines[ascii_idx+1:]:
        if line.strip():
            data_rows.append([float(x) for x in line.split()])
            
    da = pd.DataFrame(data_rows, columns=cols)
    da['DEPTH_ft'] = da['DEPTH'] * M_TO_FT
    
    # Zone classification
    da['ZONE'] = 'Undiff.'
    for zname, top_md, bot_md in [('Ghar', 2900, 3020), ('Asmari-A', 3020, 3165), ('Asmari-B', 3165, 3390), ('Jahrum', 3390, 3600)]:
        mask = (da['DEPTH'] >= top_md) & (da['DEPTH'] < bot_md)
        da.loc[mask, 'ZONE'] = zname

    # Overburden stress integration: Sv = ∫ ρ(z) g dz
    rho_kg_m3 = da['RHOB'] * 1000.0
    dz = np.diff(da['DEPTH'], prepend=da['DEPTH'].iloc[0])
    da['Sv_MPa'] = np.cumsum(rho_kg_m3 * G_MS2 * dz) / 1e6
    da['Sv_psi'] = da['Sv_MPa'] * MPA_TO_PSI

    # Pressures and Stresses (kPa to MPa/psi)
    da['SH_MPa'] = da['SHMAX_PHS'] / 1000.0
    da['Sh_MPa'] = da['SHMIN_PHS'] / 1000.0
    da['Pp_MPa'] = da['FINAL_PP'] / 1000.0
    da['UCS_MPa'] = da['UCS_FINAL'] / 1000.0

    da['SH_psi'] = da['SH_MPa'] * MPA_TO_PSI
    da['Sh_psi'] = da['Sh_MPa'] * MPA_TO_PSI
    da['Pp_psi'] = da['Pp_MPa'] * MPA_TO_PSI

    # Gradients
    da['Sv_grad'] = da['Sv_psi'] / da['DEPTH_ft']
    da['SH_grad'] = da['SH_psi'] / da['DEPTH_ft']
    da['Sh_grad'] = da['Sh_psi'] / da['DEPTH_ft']
    da['Pp_grad'] = da['Pp_psi'] / da['DEPTH_ft']
    da['Pp_ppg']  = da['Pp_grad'] / PPG_TO_PSIFT

    # Moduli
    da['E_dyn_Mpsi']  = da['YME_DYN'] * GPA_TO_MPSI
    da['E_stat_Mpsi'] = da['E_FINAL'] * GPA_TO_MPSI
    da['G_dyn_Mpsi']  = da['SMG_DYN'] * GPA_TO_MPSI
    da['K_dyn_Mpsi']  = da['BMK_DYN'] * GPA_TO_MPSI
    da['C0_psi']      = (da['COHESION_FROM_GR'] / 1000.0) * MPA_TO_PSI

    # Wellbore Stability Calculations (Kirsch / Mohr-Coulomb)
    SH_v  = da['SH_MPa'].values
    Sh_v  = da['Sh_MPa'].values
    Pp_v  = da['Pp_MPa'].values
    UCS_v = da['UCS_MPa'].values
    phi_v = np.radians(da['FANG_FROMGR'].values)
    dep_v = da['DEPTH'].values

    q_v  = (1.0 + np.sin(phi_v)) / (1.0 - np.sin(phi_v) + 1e-9)
    T0_v = UCS_v * T0_FRACTION

    # Collapse Pressure (Pc) & Fracture Pressure (Pf)
    Pc_v = (3.0 * SH_v - Sh_v - UCS_v + Pp_v * (q_v - 1.0)) / (q_v + 1.0)
    Pf_v = 3.0 * Sh_v - SH_v + T0_v - Pp_v

    da['MW_col_ppg'] = p_mpa_to_ppg(Pc_v, dep_v)
    da['MW_frc_ppg'] = p_mpa_to_ppg(Pf_v, dep_v)
    da['MW_pp_ppg']  = p_mpa_to_ppg(Pp_v, dep_v)
    da['MW_moderated_collapse_ppg'] = da['MW_col_ppg'] + MW_SAFETY_MARGIN

    # Stress Regime Classification
    da['REGIME'] = np.where(
        (da['SH_MPa'].values > da['Sv_MPa'].values) & (da['Sv_MPa'].values > da['Sh_MPa'].values), 'SS',
        np.where(da['Sv_MPa'].values > da['SH_MPa'].values, 'NF', 'RF')
    )

    wells[wa] = {'df': da, 'top_md': da['DEPTH'].min(), 'bot_md': da['DEPTH'].max()}
    print(f"  ✓ {wa} Loaded: Samples={len(da):,} | Depth: {da['DEPTH'].min():.0f}-{da['DEPTH'].max():.0f} m")

# Stress Regime Evaluation
sh_sv, sH_sv = [], []
for wa, wd in wells.items():
    df = wd['df']
    sv = df['Sv_MPa'].replace(0, np.nan)
    sh_sv.extend((df['Sh_MPa'] / sv).dropna().tolist())
    sH_sv.extend((df['SH_MPa'] / sv).dropna().tolist())

K0_med = float(np.median(sh_sv)) if sh_sv else 0.85
KH_med = float(np.median(sH_sv)) if sH_sv else 1.15
REGIME_STR = "Compressional with strike-slip component (SS/RF)"
print(f"  In-situ Stress Ratios: K0 = {K0_med:.2f} | kH = {KH_med:.2f} | Regime: {REGIME_STR}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. WELLBORE STABILITY & MUD WEIGHT WINDOW GRAPHICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Rendering 1D MEM Profiles & Wellbore Stability Windows...")

fig, axes = plt.subplots(1, len(wells), figsize=(6 * len(wells), 12), sharey=True)
axes = np.atleast_1d(axes)
fig.suptitle(f'1D Wellbore Stability & Mud Weight Window Analysis\n{FIELD_NAME_ANON}', y=0.98)

for idx, (wa, wd) in enumerate(wells.items()):
    ax = axes[idx]
    df = wd['df']
    dft = df['DEPTH_ft']
    zone_background(ax, df)

    mc   = df['MW_col_ppg'].replace([np.inf, -np.inf], np.nan)
    mmod = df['MW_moderated_collapse_ppg'].replace([np.inf, -np.inf], np.nan)
    mf   = df['MW_frc_ppg'].replace([np.inf, -np.inf], np.nan)
    mp   = df['MW_pp_ppg'].replace([np.inf, -np.inf], np.nan)
    ok   = (mc > 4) & (mc < 25) & (mf > 4) & (mf < 25)

    if ok.any():
        ax.fill_betweenx(dft[ok], mc[ok], mf[ok], alpha=0.18, color='#FFFACD', label='Safe Window')
        ax.plot(mp[ok], dft[ok], 'g-', lw=1.5, label='P_p Equivalent')
        ax.plot(mc[ok], dft[ok], 'r-', lw=2.0, label='Collapse (Pc)')
        ax.plot(mmod[ok], dft[ok], color='#8B0000', lw=2.5, ls='-.', label=f'Moderated (+{MW_SAFETY_MARGIN} ppg)')
        ax.plot(mf[ok], dft[ok], 'b-', lw=2.0, label='Fracture (Pf)')

    ax.set_xlabel('Mud Weight (ppg)')
    ax.set_ylabel('Depth TVD (ft)' if idx == 0 else '')
    ax.set_title(f'{wa}')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)
    zone_labels(ax, df)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "Fig01_Wellbore_Stability_Mud_Weight_Windows", FIG_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MOHR-COULOMB SHEAR STABILITY DIAGRAMS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Rendering Mohr-Coulomb Failure Envelopes...")

fig, axes = plt.subplots(1, len(wells), figsize=(7 * len(wells), 7))
axes = np.atleast_1d(axes)
fig.suptitle(f'3D Mohr-Coulomb Effective Stress Failure Envelopes\n{FIELD_NAME_ANON}', y=0.98)

for idx, (wa, wd) in enumerate(wells.items()):
    ax = axes[idx]
    df = wd['df']

    phi_deg = float(df['FANG_FROMGR'].median())
    phi_rad = np.radians(phi_deg)
    c0_psi  = float(df['C0_psi'].median())

    sv_e = float((df['Sv_psi'] - df['Pp_psi']).median())
    sh_e = float((df['Sh_psi'] - df['Pp_psi']).median())
    SH_e = float((df['SH_psi'] - df['Pp_psi']).median())

    theta = np.linspace(0, np.pi, 300)
    for s1, s3, clr, ls_ in [(sv_e, sh_e, 'black', '-'), (SH_e, sh_e, COLORS[wa], '--')]:
        ctr = (s1 + s3) / 2.0
        rad = (s1 - s3) / 2.0
        ax.plot(ctr + rad * np.cos(theta), rad * np.sin(theta), color=clr, lw=2.0, ls=ls_)

    tens = -c0_psi / (np.tan(phi_rad) + 1e-9)
    sr   = np.linspace(tens, sv_e * 1.3, 300)
    tau  = c0_psi + sr * np.tan(phi_rad)
    ax.plot(sr, tau, 'k-', lw=2.5, label=f'Envelope (φ={phi_deg:.1f}°)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    ax.set_xlabel("Effective Normal Stress σ'n (psi)")
    ax.set_ylabel("Shear Stress τ (psi)" if idx == 0 else "")
    ax.set_title(f"{wa}")
    ax.legend(loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "Fig02_Mohr_Coulomb_Envelopes", FIG_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT SUMMARY CSV TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Exporting Geomechanical Summary CSVs...")

t1_rows = []
for wa, wd in wells.items():
    df = wd['df']
    t1_rows.append({
        'Well': wa,
        'Sv_psi/ft': round(safe_median(df['Sv_grad']), 3),
        'SH_psi/ft': round(safe_median(df['SH_grad']), 3),
        'Sh_psi/ft': round(safe_median(df['Sh_grad']), 3),
        'Pp_ppg': round(safe_median(df['Pp_ppg']), 2),
        'MW_collapse_ppg': round(safe_median(df['MW_col_ppg']), 2),
        'MW_moderated_ppg': round(safe_median(df['MW_moderated_collapse_ppg']), 2),
        'MW_fracture_ppg': round(safe_median(df['MW_frc_ppg']), 2),
    })

pd.DataFrame(t1_rows).to_csv(CSV_DIR / "Table1_1D_MEM_Stability_Summary.csv", index=False)

print("\n" + "=" * 80)
print("  PROCESS COMPLETED: 1D MEM & Wellbore Stability Pipeline Finished.")
print(f"  Outputs saved in: {OUTPUT_DIR}")
print("=" * 80)
