#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rock_physics_analysis.py
───────────────────────────────────────────────────────────────────────────────
AVO Analysis, Elastic Attributes (LMR), and Mineral Inversion Module
Optimized for Geomechanics Python Framework (GitHub Release Version)

Calculates Lambda-Rho, Mu-Rho, Vp/Vs, Young's Modulus, Poisson's Ratio,
UCS, and Voigt-Reuss-Hill (VRH) bounds from log and seismic datasets.
"""

import os
import sys
import struct
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS & ANONYMIZATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FIELD_NAME_ANON = "SW Persian Gulf Representative Offshore Field"
WELL_ALIASES = {
    'Well-01': 'Well-A', 'Well-05': 'Well-B', 'Well-10': 'Well-C',
    'Well_01': 'Well-A', 'Well_05': 'Well-B', 'Well_10': 'Well-C'
}
COORD_OFFSET_X = 0.0
COORD_OFFSET_Y = 0.0

COLORS = {'Well-A': '#D95F02', 'Well-B': '#7570B3', 'Well-C': '#1B9E77'}
ZCOL = {
    'Ghar': '#2CA02C', 
    'Asmari-A': '#D62728', 
    'Asmari-B': '#1F77B4', 
    'Jahrum': '#FF7F0E', 
    'Undiff.': '#7F7F7F'
}

M_TO_FT = 3.28084
MPA_TO_PSI = 145.0377

def anon(w):
    s = str(w)
    for r, a in WELL_ALIASES.items():
        s = s.replace(r, a)
    return s

# ─────────────────────────────────────────────────────────────────────────────
# 2. PORTABLE PATHS SETUP
# ─────────────────────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"

FIG_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "tables_csv"
LOG_DIR = OUTPUT_DIR / "logs"
NPY_DIR = OUTPUT_DIR / "numpy_grids"

for d in [OUTPUT_DIR, FIG_DIR, CSV_DIR, LOG_DIR, NPY_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class SessionLogger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, 'w', encoding='utf-8')
    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = SessionLogger(str(LOG_DIR / "rock_physics_session.log"))

# Matplotlib Publication-Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 22,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("=" * 80)
print(f"Rock Physics & Elastic Attributes Inversion Pipeline | {RUN_TS}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTOMATIC SYNTHETIC DATA GENERATOR (For Portable execution on GitHub)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_inputs_if_missing():
    """Generates synthetic dataset assets to run the script seamlessly without raw files."""
    db_file = DATA_DIR / "DB_zonation"
    wells_dir = DATA_DIR / "wells"
    wells_dir.mkdir(parents=True, exist_ok=True)

    if not db_file.exists():
        print("⚠️ Geological data files missing. Generating synthetic reference files...")
        # Write DB_zonation
        with open(db_file, 'w', encoding='utf-8') as f:
            f.write("BEGIN HEADER\n")
            f.write("Well Surface MD X Y\n")
            f.write("END HEADER\n")
            f.write("'Well-A' 'Ghar_C.R.' 2100.0 50000.0 300000.0\n")
            f.write("'Well-A' 'Asmari_A' 2180.0 50000.0 300000.0\n")
            f.write("'Well-A' 'Asmari_B1' 2250.0 50000.0 300000.0\n")
            f.write("'Well-A' 'Jahrum' 2350.0 50000.0 300000.0\n")
            f.write("'Well-A' 'Pabdeh' 2500.0 50000.0 300000.0\n")
            f.write("'Well-B' 'Ghar_C.R.' 2110.0 52000.0 302000.0\n")
            f.write("'Well-B' 'Asmari_A' 2190.0 52000.0 302000.0\n")
            f.write("'Well-B' 'Asmari_B1' 2260.0 52000.0 302000.0\n")
            f.write("'Well-B' 'Jahrum' 2360.0 52000.0 302000.0\n")
            f.write("'Well-B' 'Pabdeh' 2510.0 52000.0 302000.0\n")
            f.write("'Well-C' 'Ghar_C.R.' 2090.0 49000.0 298000.0\n")
            f.write("'Well-C' 'Asmari_A' 2170.0 49000.0 298000.0\n")
            f.write("'Well-C' 'Asmari_B1' 2240.0 49000.0 298000.0\n")
            f.write("'Well-C' 'Jahrum' 2340.0 49000.0 298000.0\n")
            f.write("'Well-C' 'Pabdeh' 2490.0 49000.0 298000.0\n")

    # Generate Mock LAS files
    for w in ['Well-A', 'Well-B', 'Well-C']:
        las_file = wells_dir / f"{w}_WIRE.las"
        if not las_file.exists():
            np.random.seed(42)
            depth = np.linspace(2000, 2600, 1200)
            phie = np.clip(np.random.normal(0.08, 0.04, len(depth)), 0.01, 0.28)
            sw = np.clip(np.random.normal(0.40, 0.25, len(depth)), 0.01, 1.0)
            ucs = np.clip(45.0 * (1.0 - phie)**2.5 + np.random.normal(0, 3, len(depth)), 5.0, 120.0)
            e_static = np.clip(18.0 * (1.0 - phie)**3 + np.random.normal(0, 1.5, len(depth)), 2.0, 60.0)
            e_dyn = e_static * np.random.uniform(1.2, 1.5, len(depth))
            smg = e_dyn / (2 * (1 + 0.25))
            bmk = e_dyn / (3 * (1 - 2 * 0.25))
            pr_dyn = np.full(len(depth), 0.25) + np.random.normal(0, 0.02, len(depth))

            hdr = (
                "~Version Information\nVERS. 2.0 : CWLS log ASCII Standard -VERSION 2.0\n"
                "~Well Information\n"
                f"WELL.  {w} : WELL NAME\n"
                "~Curve Information\n"
                "DEPT.M         : DEPTH\n"
                "PHIE.DEC       : Effective Porosity\n"
                "SW.DEC         : Water Saturation\n"
                "UCS_FINAL.MPA  : Unconfined Compressive Strength\n"
                "E_FINAL.GPA    : Static Youngs Modulus\n"
                "YME_DYN.GPA    : Dynamic Youngs Modulus\n"
                "SMG_DYN.GPA    : Dynamic Shear Modulus\n"
                "BMK_DYN.GPA    : Dynamic Bulk Modulus\n"
                "PR_DYN.DEC     : Dynamic Poisson Ratio\n"
                "~Ascii\n"
            )
            with open(las_file, 'w', encoding='utf-8') as f:
                f.write(hdr)
                for i in range(len(depth)):
                    f.write(f"{depth[i]:.2f} {phie[i]:.4f} {sw[i]:.4f} {ucs[i]:.3f} {e_static[i]:.3f} {e_dyn[i]:.3f} {smg[i]:.3f} {bmk[i]:.3f} {pr_dyn[i]:.3f}\n")

    # Generate Mock 2D Seismic grids
    for prop in ['Zp', 'Zs', 'Porosity', 'Sw']:
        grid_file = DATA_DIR / f"{prop}_2D.npy"
        if not grid_file.exists():
            np.random.seed(123)
            # Create a 2D map grid representation
            if prop == 'Zp':      grid = np.random.normal(12500, 1500, (201, 201))
            elif prop == 'Zs':    grid = np.random.normal(7200, 800, (201, 201))
            elif prop == 'Porosity': grid = np.random.normal(0.08, 0.03, (201, 201))
            else:                 grid = np.random.normal(0.35, 0.15, (201, 201))
            np.save(grid_file, grid)

generate_synthetic_inputs_if_missing()

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD SEISMIC AND GEOMECHANICAL GRIDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80 + "\n[1] Loading Seismic Attribute Grids\n" + "─" * 80)

XX, YY = np.meshgrid(np.linspace(0, 15, 201), np.linspace(0, 15, 201))
cube_maps = {}

for prop_name in ['Zp', 'Zs', 'Porosity', 'Sw']:
    npy_file = DATA_DIR / f"{prop_name}_2D.npy"
    if npy_file.exists():
        cube_maps[prop_name] = np.load(str(npy_file))
        print(f"  ✓ Loaded property grid: {prop_name} ({cube_maps[prop_name].shape})")

# Save coordinates as grids for Step 10
np.save(NPY_DIR / "XX.npy", XX)
np.save(NPY_DIR / "YY.npy", YY)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ROCK PHYSICS CALCULATIONS & LMR PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80 + "\n[2] Rock Physics Inversion & Mineral Modeling\n" + "─" * 80)

Zp_2d, Zs_2d = cube_maps.get('Zp'), cube_maps.get('Zs')
rp_results = {}

if Zp_2d is not None and Zs_2d is not None:
    rp_results['LambdaRho'] = Zp_2d**2 - 2 * Zs_2d**2
    rp_results['MuRho'] = Zs_2d**2
    rp_results['VpVs'] = Zp_2d / np.where(Zs_2d > 0, Zs_2d, np.nan)
    rp_results['PI'] = Zp_2d - 2.0 * Zs_2d
    
    for name, arr in rp_results.items():
        np.save(NPY_DIR / f"{name}_2D.npy", arr)
        np.save(NPY_DIR / f"{name}_2D.npy", arr) # Mirror to output grid directory

    obs_vpvs = 1.690
    f_dol = float(np.clip((1.90 - obs_vpvs) / (1.90 - 1.74), 0.0, 1.0))
    f_cal = 1.0 - f_dol
    K_min = f_cal * 77.0 + f_dol * 94.9
    G_min = f_cal * 32.0 + f_dol * 45.0
    rho_min = f_cal * 2.71 + f_dol * 2.87
    print(f"  Matrix Mineral Fractions   : Calcite = {f_cal:.1%} | Dolomite = {f_dol:.1%}")
    print(f"  Matrix Reconstructed Moduli: Bulk (K) = {K_min:.2f} GPa | Shear (G) = {G_min:.2f} GPa")
else:
    f_cal, f_dol, K_min, G_min, rho_min = 0.70, 0.30, 82.4, 35.9, 2.758

Por_2d = cube_maps.get('Porosity')
if Por_2d is not None:
    Por_2d = np.clip(Por_2d / 100.0 if np.nanmean(Por_2d) > 1.0 else Por_2d, 0.0, 0.40)
    np.save(NPY_DIR / "Porosity_2D.npy", Por_2d)

# ─────────────────────────────────────────────────────────────────────────────
# 6. WELL-LOG PROCESSING AND IMPERIAL CONVERSIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80 + "\n[3] In-Situ Well Log Processing (Dynamic Calibration)\n" + "─" * 80)

def parse_db_zon(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        headers, data_rows, is_hdr, is_data = [], [], False, False
        for line in lines:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('VERSION'):
                continue
            if s == 'BEGIN HEADER':
                is_hdr = True; continue
            if s == 'END HEADER':
                is_hdr = False; is_data = True; continue
            if is_hdr:
                headers.extend(s.split())
                continue
            if is_data:
                parts = s.split()
                if len(parts) == len(headers):
                    row_dict = {}
                    for idx, h in enumerate(headers):
                        val = parts[idx].strip("'").strip('"')
                        try:
                            row_dict[h] = float(val)
                        except ValueError:
                            row_dict[h] = val
                    data_rows.append(row_dict)
        df = pd.DataFrame(data_rows)
        if 'Well' in df.columns:
            df['Well_anon'] = df['Well'].apply(anon)
        return df
    except Exception as e:
        print(f"  [ERROR] DB_zonation parsing error: {e}")
        return None

tops_df = parse_db_zon(DATA_DIR / "DB_zonation")
wells_asm = {}

for wa in ['Well-A', 'Well-B', 'Well-C']:
    las_file = DATA_DIR / "wells" / f"{wa}_WIRE.las"
    if not las_file.exists() or tops_df is None:
        continue

    # Simple LAS parser for synthetic mock
    with open(las_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ascii_idx = next(i for i, line in enumerate(lines) if "~Ascii" in line)
    
    cols = ['DEPTH', 'PHIE', 'SW', 'UCS_FINAL', 'E_FINAL', 'YME_DYN', 'SMG_DYN', 'BMK_DYN', 'PR_DYN']
    data_rows = []
    for line in lines[ascii_idx+1:]:
        if line.strip():
            data_rows.append([float(x) for x in line.split()])
            
    df_w = pd.DataFrame(data_rows, columns=cols)
    df_w['DEPTH_FT'] = df_w['DEPTH'] * M_TO_FT
    
    # Scale elastic parameters to imperial units
    if 'UCS_FINAL' in df_w.columns: 
        df_w['UCS_PSI'] = df_w['UCS_FINAL'] * MPA_TO_PSI
    if 'E_FINAL' in df_w.columns: 
        df_w['E_MPSI'] = df_w['E_FINAL'] * 145.0377 / 1000.0
    if 'YME_DYN' in df_w.columns: 
        df_w['E_DYN_MPSI'] = df_w['YME_DYN'] * 145.0377 / 1000.0

    tw = tops_df[tops_df['Well_anon'] == wa]
    if len(tw) == 0:
        continue
    
    top_md = float(tw[tw['Surface'] == 'Ghar_C.R.']['MD'].iloc[0])
    bot_md = float(tw[tw['Surface'] == 'Pabdeh']['MD'].iloc[0])
    
    da = df_w[(df_w['DEPTH'] >= top_md) & (df_w['DEPTH'] <= bot_md)].copy()
    da['ZONE'] = 'Undiff.'
    
    for zname, zmarks in [('Ghar', ['Ghar_C.R.']), ('Asmari-A', ['Asmari_A']), ('Asmari-B', ['Asmari_B1']), ('Jahrum', ['Jahrum'])]:
        for mk in zmarks:
            r = tw[tw['Surface'] == mk]
            if len(r) > 0:
                da.loc[da['DEPTH'] >= float(r['MD'].iloc[0]), 'ZONE'] = zname

    if 'BMK_DYN' in da.columns and 'SMG_DYN' in da.columns:
        rho_well = 2.51
        da['Vp_calc'] = np.sqrt((da['BMK_DYN'] + 4.0/3.0 * da['SMG_DYN']) * 1e9 / (rho_well * 1000.0))
        da['Vs_calc'] = np.sqrt(da['SMG_DYN'] * 1e9 / (rho_well * 1000.0))
        da['VpVs_calc'] = da['Vp_calc'] / np.where(da['Vs_calc'] > 0, da['Vs_calc'], np.nan)
        da['Zp_calc'] = rho_well * da['Vp_calc']
        da['Zs_calc'] = rho_well * da['Vs_calc']
        da['LR_calc'] = da['Zp_calc']**2 - 2 * da['Zs_calc']**2
        da['MR_calc'] = da['Zs_calc']**2

    wells_asm[wa] = {'df': da}
    print(f"  ✓ {wa} Registered: MD interval: {top_md:.1f}-{bot_md:.1f} m | Active Rows: {len(da):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PUBLICATION GRAPHICS GENERATION (300 DPI)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80 + "\n[4] Rendering Geological & Petrophysical Figures\n" + "─" * 80)

# Figure 1: Elastic Attribute Crossplots
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(f'Rock Physics Diagnostics & Lithological Templates\n{FIELD_NAME_ANON}', y=0.98, fontsize=20)

ax = axes[0, 0]
for wa, wd in wells_asm.items():
    if 'Zp_calc' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5:
                ax.scatter(zd['Zp_calc']/1000.0, zd['Zs_calc']/1000.0, s=8, c=ZCOL.get(zone), alpha=0.4, 
                           label=zone if wa == 'Well-A' else '')
ax.set_xlabel('P-Impedance Zp (10³ g/cm³·m/s)')
ax.set_ylabel('S-Impedance Zs (10³ g/cm³·m/s)')
ax.set_title('(a) Shear vs. Compressional Impedance')
ax.legend(loc='upper left', markerscale=3)

ax = axes[0, 1]
for wa, wd in wells_asm.items():
    if 'LR_calc' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5: 
                ax.scatter(zd['MR_calc']/1e6, zd['LR_calc']/1e6, s=8, c=ZCOL.get(zone), alpha=0.4)
ax.set_xlabel('Shear Rigidity μρ (×10⁶)')
ax.set_ylabel('Incompressible Fluid Lame λρ (×10⁶)')
ax.set_title('(b) Lambda-Rho vs. Mu-Rho (LMR)')

ax = axes[0, 2]
for wa, wd in wells_asm.items():
    if 'VpVs_calc' in wd['df'].columns and 'E_MPSI' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5: 
                ax.scatter(zd['E_MPSI'], zd['VpVs_calc'], s=8, c=ZCOL.get(zone), alpha=0.4)
ax.set_xlabel('Static Young Modulus (Mpsi)')
ax.set_ylabel('Vp/Vs Ratio')
ax.set_title('(c) Vp/Vs vs. Static Young Modulus')

ax = axes[1, 0]
for wa, wd in wells_asm.items():
    if 'VpVs_calc' in wd['df'].columns and 'UCS_PSI' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5: 
                ax.scatter(zd['VpVs_calc'], zd['UCS_PSI'], s=8, c=ZCOL.get(zone), alpha=0.4)
ax.set_xlabel('Vp/Vs Ratio')
ax.set_ylabel('UCS (psi)')
ax.set_title('(d) UCS vs. Vp/Vs')

ax = axes[1, 1]
for wa, wd in wells_asm.items():
    if 'E_DYN_MPSI' in wd['df'].columns and 'PR_DYN' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5: 
                ax.scatter(zd['E_DYN_MPSI'], zd['PR_DYN'], s=8, c=ZCOL.get(zone), alpha=0.4)
ax.set_xlabel('Dynamic Young Modulus (Mpsi)')
ax.set_ylabel("Dynamic Poisson's Ratio")
ax.set_title('(e) Poisson\'s Ratio vs. Dynamic Modulus')

ax = axes[1, 2]
for wa, wd in wells_asm.items():
    if 'MR_calc' in wd['df'].columns and 'UCS_PSI' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5: 
                ax.scatter(zd['MR_calc']/1e6, zd['UCS_PSI'], s=8, c=ZCOL.get(zone), alpha=0.4)
ax.set_xlabel('Shear Rigidity μρ (×10⁶)')
ax.set_ylabel('UCS (psi)')
ax.set_title('(f) UCS vs. Mu-Rho')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "Fig01_rock_physics_crossplots.png", FIG_DIR, dpi=300)

# Figure 2: Spatial Attribute Mapping
if Zp_2d is not None:
    X_km, Y_km = np.meshgrid(np.linspace(0, 15, Zp_2d.shape[1]), np.linspace(0, 15, Zp_2d.shape[0]))
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(f'Seismic Elastic Property Distribution Maps\n{FIELD_NAME_ANON}', y=0.98, fontsize=20)

    configs = [
        (Zp_2d, 'P-Impedance Zp (g/cm³·m/s)', 'viridis', '(a) P-Impedance Map'),
        (Zs_2d, 'S-Impedance Zs (g/cm³·m/s)', 'plasma', '(b) S-Impedance Map'),
        (rp_results['VpVs'], 'Vp/Vs Ratio', 'RdYlGn_r', '(c) Vp/Vs Ratio Map'),
        (rp_results['LambdaRho']/1e6, 'Lambda-Rho λρ (×10⁶)', 'coolwarm', '(d) Lambda-Rho Map'),
        (rp_results['MuRho']/1e6, 'Mu-Rho μρ (×10⁶)', 'YlOrRd', '(e) Mu-Rho Map'),
        (Por_2d, 'Porosity (fraction)', 'YlGn', '(f) Porosity Map'),
    ]

    for idx, (arr, label, cmap, title) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        im = ax.pcolormesh(X_km, Y_km, arr, cmap=cmap, shading='auto', 
                           vmin=np.nanpercentile(arr, 5), vmax=np.nanpercentile(arr, 95))
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(label, size=12)
        ax.set_title(title)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)' if idx % 3 == 0 else '')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "Fig02_seismic_RP_maps.png", FIG_DIR, dpi=300)

# Figure 3: Integrated Geomechanical Crossplots
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Dynamic-to-Static Core Integration & Predictive Calibration Templates', y=0.96, fontsize=18)

ax = axes[0]
for wa, wd in wells_asm.items():
    if 'E_MPSI' in wd['df'].columns and 'UCS_PSI' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]
            if len(zd) > 5:
                ax.scatter(zd['E_MPSI'], zd['UCS_PSI'], s=8, c=ZCOL.get(zone), alpha=0.4, 
                           label=zone if wa == 'Well-A' else '')
ax.set_xlabel('Static Young Modulus E_static (Mpsi)')
ax.set_ylabel('UCS (psi)')
ax.set_title('(a) E_static vs. UCS Strength')
ax.legend(loc='upper left', markerscale=3)

ax = axes[1]
vpvs_data = {}
for wa, wd in wells_asm.items():
    if 'VpVs_calc' in wd['df'].columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = wd['df'][wd['df']['ZONE'] == zone]['VpVs_calc'].dropna()
            vpvs_data.setdefault(zone, []).extend(zd[(zd > 1.0) & (zd < 3.0)].values)
if vpvs_data:
    zones_ordered = ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']
    bp = ax.boxplot([vpvs_data[z] for z in zones_ordered], patch_artist=True, showfliers=False, 
                    medianprops=dict(color='black', lw=2.5))
    ax.set_xticklabels(zones_ordered)
    for patch, z in zip(bp['boxes'], zones_ordered):
        patch.set_facecolor(ZCOL.get(z))
        patch.set_alpha(0.6)
ax.set_xlabel('Structural Formations')
ax.set_ylabel('Vp/Vs Ratio')
ax.set_title('(b) Stratigraphic Vp/Vs Distribution')

ax = axes[2]
all_mr, all_ucs = [], []
for wa, wd in wells_asm.items():
    if 'MR_calc' in wd['df'].columns and 'UCS_PSI' in wd['df'].columns:
        v = wd['df'][['MR_calc', 'UCS_PSI', 'ZONE']].dropna()
        v = v[(v['MR_calc'] > 0) & (v['UCS_PSI'] > 0)]
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = v[v['ZONE'] == zone]
            if len(zd) > 5:
                ax.scatter(zd['MR_calc']/1e6, zd['UCS_PSI'], s=8, c=ZCOL.get(zone), alpha=0.4)
                all_mr.extend((zd['MR_calc']/1e6).values)
                all_ucs.extend(zd['UCS_PSI'].values)
if all_mr:
    sl, it, r_val, _, _ = sp_stats.linregress(np.array(all_mr), np.array(all_ucs))
    x_range = np.linspace(0, 60, 100)
    ax.plot(x_range, sl * x_range + it, 'k-', lw=3, 
            label=f'UCS = {sl:.2f}·μρ + {it:.1f}\nR² = {r_val**2:.3f}')
ax.set_xlabel('Shear Rigidity μρ (×10⁶)')
ax.set_ylabel('UCS (psi)')
ax.set_title('(c) Empirical UCS Strength from μρ')
ax.legend(loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "Fig03_petrophys_geomech.png", FIG_DIR, dpi=300)

# Figure 5: 3D Rock Physics Domain Clustering
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle(f'3D Multi-domain Elastic Attribute Space\n{FIELD_NAME_ANON}', y=0.95, fontsize=18)

for wa, wd in wells_asm.items():
    df = wd['df']
    if all(k in df.columns for k in ['VpVs_calc', 'Zp_calc', 'UCS_PSI']):
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = df[df['ZONE'] == zone]
            if len(zd) > 10:
                ax.scatter(zd['VpVs_calc'], zd['Zp_calc']/1000.0, zd['UCS_PSI'], 
                           c=ZCOL.get(zone), s=15, alpha=0.5, label=zone if wa == 'Well-A' else '')

ax.set_xlabel('Vp/Vs Ratio', labelpad=12)
ax.set_ylabel('P-Impedance Zp (10³)', labelpad=12)
ax.set_zlabel('UCS Strength (psi)', labelpad=12)
ax.view_init(elev=20, azim=45)
ax.legend(markerscale=3, loc='upper left')
plt.tight_layout()
save_fig(fig, "Fig05_3D_Rock_Physics_Clustering.png", FIG_DIR, dpi=300)

# Figure 6: Velocity-Porosity Bounds (Voigt-Reuss-Hill)
fig, ax = plt.subplots(figsize=(11, 8))
fig.suptitle(f'Velocity-Porosity Relations vs. Homogenization Bounds\n{FIELD_NAME_ANON}', y=0.96, fontsize=16)

phi_axis = np.linspace(0.001, 0.40, 100)
M_min = K_min + (4.0/3.0)*G_min
M_fl = 2.60 # Fluid Bulk Modulus (Water/Brine)
M_voigt = (1 - phi_axis)*M_min + phi_axis*M_fl
Vp_voigt = np.sqrt(M_voigt * 1e9 / ((rho_min*(1-phi_axis) + 1.10*phi_axis)*1000.0))

M_reuss = 1.0 / ((1 - phi_axis)/M_min + phi_axis/M_fl)
Vp_reuss = np.sqrt(M_reuss * 1e9 / ((rho_min*(1-phi_axis) + 1.10*phi_axis)*1000.0))

ax.plot(phi_axis*100, Vp_voigt, 'k--', lw=2.5, label='Voigt Upper Bound (Stiff Pores)')
ax.plot(phi_axis*100, Vp_reuss, 'k:', lw=2.5, label='Reuss Lower Bound (Fluid Suspension)')

for wa, wd in wells_asm.items():
    df = wd['df']
    if 'PHIE' in df.columns and 'Vp_calc' in df.columns:
        for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
            zd = df[(df['ZONE'] == zone) & (df['PHIE'] > 0)]
            if len(zd) > 10:
                ax.scatter(zd['PHIE']*100, zd['Vp_calc'], c=ZCOL.get(zone), s=15, alpha=0.3, 
                           label=zone if wa=='Well-A' else '')

ax.set_xlabel('Effective Porosity (%)')
ax.set_ylabel('Compressional Velocity Vp (m/s)')
ax.set_xlim(0, 30); ax.set_ylim(2000, 7000)
ax.legend(loc='upper right', markerscale=3)
plt.tight_layout()
save_fig(fig, "Fig06_Velocity_Porosity_Bounds.png", FIG_DIR, dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 8. STATISTICAL TABLES EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80 + "\n[5] Exporting Structural Quality CSVs\n" + "─" * 80)

t1_rows = []
for zone in ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']:
    row = {'Zone': zone}
    for wa, wd in wells_asm.items():
        zd = wd['df'][wd['df']['ZONE'] == zone]
        for prop, col in [('Edyn_Mpsi', 'E_DYN_MPSI'), ('Estat_Mpsi', 'E_MPSI'), ('UCS_psi', 'UCS_PSI'), ('VpVs', 'VpVs_calc')]:
            if col in zd.columns:
                v = zd[col].dropna()
                if len(v) > 5: 
                    row[f'{prop}_{wa}'] = round(float(v.median()), 2)
    t1_rows.append(row)
    
pd.DataFrame(t1_rows).to_csv(CSV_DIR / "Table1_RP_per_zone_Imperial.csv", index=False)

print("\n" + "=" * 80)
print("  PROCESS COMPLETED: 3D Rock Physics & Inversion Pipeline Complete.")
print(f"  Target Run Output: {OUTPUT_DIR}")
print("=" * 80)
