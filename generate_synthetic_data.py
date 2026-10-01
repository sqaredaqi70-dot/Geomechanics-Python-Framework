#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
                  SYNTHETIC WELL DATA GENERATOR
═══════════════════════════════════════════════════════════════════════
Generates physically consistent synthetic LAS files and formation tops
for demonstration of the Geomechanics Python Framework.

All generated data is 100% synthetic and computer-generated.
It does not represent any real subsurface oil/gas asset.

Author:   Saeed Gharedaghi
Contact:  sqaredaqi70@gmail.com
License:  AGPL-3.0
═══════════════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import lasio
except ImportError:
    print("ERROR: 'lasio' package not found.")
    print("Install with: pip install lasio")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DIRECTORY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "data" / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also write directly to ucs_calibration and dynamic_to_static modules
UCS_DATA_DIR = ROOT_DIR / "modules" / "ucs_calibration" / "data"
D2S_DATA_DIR = ROOT_DIR / "modules" / "dynamic_to_static_calibration" / "data"
FR_DATA_DIR = ROOT_DIR / "modules" / "fault_reactivation" / "data"
RP_DATA_DIR = ROOT_DIR / "modules" / "rock_physics" / "data"

for d in [UCS_DATA_DIR, D2S_DATA_DIR, FR_DATA_DIR, RP_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WELLS = {
    'Well_A': {'top': 2900.0, 'bot': 3550.0, 'seed': 42},
    'Well_B': {'top': 2950.0, 'bot': 3600.0, 'seed': 123},
    'Well_C': {'top': 3050.0, 'bot': 3700.0, 'seed': 456},
}

ZONE_OFFSETS = {
    'Ghar_C.R.':   0.0,
    'Ghar_1':      35.0,
    'Ghar_2':      70.0,
    'Ghar_3_1':    95.0,
    'Asmari_A':    120.0,
    'AA_Shale':    240.0,
    'Asmari_B1':   265.0,
    'Asmari_B2_1': 340.0,
    'Asmari_B3':   420.0,
    'Jahrum':      490.0,
    'Pabdeh':      640.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHYSICAL SIMULATION ENGINES
# ─────────────────────────────────────────────────────────────────────────────
def generate_gr(depth, seed=42):
    rng = np.random.default_rng(seed)
    base = 25 + 15 * np.sin(depth * 0.003)
    noise = rng.normal(0, 8, len(depth))
    shale_events = rng.random(len(depth)) < 0.05
    gr = base + noise + shale_events * rng.uniform(40, 80, len(depth))
    return np.clip(gr, 10, 200)

def generate_density(depth, seed=42):
    rng = np.random.default_rng(seed)
    compaction = 2.35 + 0.00015 * depth
    noise = rng.normal(0, 0.05, len(depth))
    porosity_effect = -0.15 * np.sin(depth * 0.005)
    rho = compaction + noise + porosity_effect
    return np.clip(rho, 2.0, 2.85)

def generate_sonic_p(depth, seed=42):
    rng = np.random.default_rng(seed)
    base = 65 - 0.003 * depth
    noise = rng.normal(0, 4, len(depth))
    dt = base + noise + 8 * np.sin(depth * 0.004)
    return np.clip(dt, 45, 110)

def generate_sonic_s(dt_p, seed=42):
    rng = np.random.default_rng(seed)
    vp_vs_ratio = 1.85
    noise = rng.normal(0, 3, len(dt_p))
    return dt_p * vp_vs_ratio + noise

def generate_ucs(depth, gr, seed=42):
    rng = np.random.default_rng(seed)
    ucs_base_mpa = 50 + 30 * np.sin(depth * 0.003)
    shale_penalty = np.where(gr > 60, -20, 0)
    ucs_mpa = np.clip(ucs_base_mpa + shale_penalty + rng.normal(0, 10, len(depth)), 5, 150)
    return ucs_mpa * 1000

def generate_stresses(depth, seed=42):
    rng = np.random.default_rng(seed)
    sv_mpa = 22.6 * depth / 1000.0
    sHmax_mpa = sv_mpa * (1.15 + rng.normal(0, 0.03, len(depth)))
    shmin_mpa = sv_mpa * (0.85 + rng.normal(0, 0.02, len(depth)))
    return sv_mpa * 1000.0, sHmax_mpa * 1000.0, shmin_mpa * 1000.0

def generate_pore_pressure(depth, seed=42):
    rng = np.random.default_rng(seed)
    pp_mpa = 10.5 * depth / 1000.0 + rng.normal(0, 0.3, len(depth))
    return pp_mpa * 1000.0

def generate_moduli(depth, dt_p, dt_s, rho_g_cc, seed=42):
    """Calculates dynamically consistent elastic moduli using wave propagation equations."""
    rng = np.random.default_rng(seed)
    
    # Velocities conversion (us/ft -> m/s)
    vp = 304800.0 / dt_p
    vs = 304800.0 / dt_s
    rho_kg_m3 = rho_g_cc * 1000.0
    
    # Dynamic moduli calculations
    g_dyn_gpa = (rho_kg_m3 * vs**2) * 1e-9
    k_dyn_gpa = (rho_kg_m3 * (vp**2 - (4.0/3.0)*vs**2)) * 1e-9
    e_dyn_gpa = (9.0 * k_dyn_gpa * g_dyn_gpa) / (3.0 * k_dyn_gpa + g_dyn_gpa)
    pr_dyn = (3.0 * k_dyn_gpa - 2.0 * g_dyn_gpa) / (6.0 * k_dyn_gpa + 2.0 * g_dyn_gpa)
    
    # Core calibration correlations (static conversions)
    e_sta_gpa = 0.8468 * e_dyn_gpa + 1.221 + rng.normal(0, 2.0, len(depth))
    pr_sta = pr_dyn * 0.90
    
    return (np.clip(e_dyn_gpa, 10, 110), 
            np.clip(e_sta_gpa, 5, 95), 
            np.clip(pr_dyn, 0.15, 0.45), 
            np.clip(pr_sta, 0.15, 0.40))

def generate_friction_cohesion(gr, ucs_kpa, seed=42):
    rng = np.random.default_rng(seed)
    fang = 37.6 - 0.15 * gr + rng.normal(0, 2.5, len(gr))
    fang = np.clip(fang, 15, 45)
    cohesion_kpa = ucs_kpa * 0.16 + rng.normal(0, 400, len(gr))
    return fang, np.clip(cohesion_kpa, 100, 20000)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SAMPLE PRODUCTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def create_synthetic_las(well_name, well_config, out_path):
    seed = well_config['seed']
    top = well_config['top']
    bot = well_config['bot']

    # 0.1524m sampling rate
    depth = np.arange(top, bot + 0.1524, 0.1524)
    n = len(depth)

    gr = generate_gr(depth, seed)
    rho = generate_density(depth, seed)
    dt_p = generate_sonic_p(depth, seed)
    dt_s = generate_sonic_s(dt_p, seed)
    ucs_kpa = generate_ucs(depth, gr, seed)
    sv_kpa, sHmax_kpa, shmin_kpa = generate_stresses(depth, seed)
    pp_kpa = generate_pore_pressure(depth, seed)
    e_dyn, e_sta, pr_dyn, pr_sta = generate_moduli(depth, dt_p, dt_s, rho, seed)
    fang, cohesion_kpa = generate_friction_cohesion(gr, ucs_kpa, seed)
    tstr_kpa = ucs_kpa * 0.10

    df = pd.DataFrame({
        'DEPT':          depth,
        'GR':            gr,
        'DT':            dt_p,
        'DTS':           dt_s,
        'RHOB':          rho,
        'UCS_FINAL':     ucs_kpa,
        'UCS_FINAL_70':  ucs_kpa * 0.70,
        'UCS_FINAL_80':  ucs_kpa * 0.80,
        'UCS_FINAL_90':  ucs_kpa * 0.90,
        'UCS_FINAL_110': ucs_kpa * 1.10,
        'UCS_FINAL_120': ucs_kpa * 1.20,
        'UCS_HORSRUD':   ucs_kpa * (1.0 + np.random.normal(0, 0.08, n)),
        'UCS_MCNALLY':   ucs_kpa * (1.0 + np.random.normal(0, 0.11, n)),
        'UCS_CDE':       ucs_kpa * (1.0 + np.random.normal(0, 0.14, n)),
        'UCS_SMG_RPC':   ucs_kpa * (1.0 + np.random.normal(0, 0.12, n)),
        'UCS_SND_RPC':   ucs_kpa * (1.0 + np.random.normal(0, 0.13, n)),
        'UCS_YME':       ucs_kpa * (1.0 + np.random.normal(0, 0.10, n)),
        'TSTR_10%':      tstr_kpa,
        'TSTR_8_4%':     tstr_kpa * 0.84,
        'SHMAX_PHS':     sHmax_kpa,
        'SHMAX_MC_UB':   sHmax_kpa * 1.05,
        'SHMIN_PHS':     shmin_kpa,
        'SHMIN_MC_LB':   shmin_kpa * 0.95,
        'FINAL_PP':      pp_kpa,
        'FINAL_PP_COR':  pp_kpa * 1.02,
        'YME_DYN':       e_dyn,
        'E_FINAL':       e_sta,
        'YME_STA_HMC':   e_sta * 1.05,
        'YME_STA_JFC':   e_sta * 0.98,
        'YME_STA_MMC':   e_sta * 1.02,
        'YME_STA_PBC':   e_sta * 0.95,
        'PR_DYN':        pr_dyn,
        'PR_STA':        pr_sta,
        'SMG_DYN':       e_dyn / (2 * (1 + pr_dyn)),
        'BMK_DYN':       e_dyn / (3 * (1 - 2 * pr_dyn)),
        'FANG_FROMGR':         fang,
        'COHESION_FROM_GR':    cohesion_kpa,
        'COHESION_FROM_PLUMB': cohesion_kpa * 0.95,
    })

    las = lasio.LASFile()
    las.well['WELL'] = well_name
    las.well['FLD']  = 'Synthetic Field'
    las.well['SRVC'] = 'Geomechanics Python Framework - Synthetic Generator v3.0'
    las.well['UWI']  = f'SYN-{well_name}'

    las.well['STRT'].value = float(depth[0])
    las.well['STOP'].value = float(depth[-1])
    las.well['STEP'].value = 0.1524
    las.well['NULL'].value = -999.25

    for col in df.columns:
        unit = 'M' if col == 'DEPT' else ''
        las.add_curve(col, df[col].values, unit=unit, descr=f'Synthetic {col}')

    las.write(str(out_path))
    return len(df), df

def create_synthetic_tops(out_path):
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        "#           SYNTHETIC WELL TOPS - GEOMECHANICS FRAMEWORK",
        "# ══════════════════════════════════════════════════════════════",
        "",
        f"{'ID':>4} {'Well':10} {'Surface':15} {'MD':>10} {'Z':>10}",
        "=" * 55,
    ]
    idx = 1
    for well_name, wcfg in WELLS.items():
        for surface, offset in ZONE_OFFSETS.items():
            md = wcfg['top'] + offset
            z = -md
            lines.append(f"{idx:>4} {well_name:10} {surface:15} {md:>10.2f} {z:>10.2f}")
            idx += 1

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ─────────────────────────────────────────────────────────────────────────────
# 4. MASTER ORCHESTRATION EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Orchestrating Global Geomechanics Synthetic Data Generation")
    print("  License: AGPL-3.0 | Educational & Verification Use Only")
    print("=" * 70)
    
    # 1. Create LAS files and gather master dataset
    master_dfs = []
    for well_name, wcfg in WELLS.items():
        # Clean naming for outputs (A, B, C instead of _A, _B, _C)
        clean_name = well_name.replace("_", "-")
        out_path = OUTPUT_DIR / f"{clean_name}_synthetic.las"
        n_rows, df_well = create_synthetic_las(clean_name, wcfg, out_path)
        
        # Mirror outputs to modules
        shutil.copy2(out_path, D2S_DATA_DIR / f"{clean_name}_synthetic.las")
        shutil.copy2(out_path, RP_DATA_DIR / f"{clean_name}_synthetic.las")
        
        # Format for ML master dataset
        df_well['Well'] = clean_name
        df_well['ZONE'] = 'Undiff.'
        for surface, offset in ZONE_OFFSETS.items():
            if 'Ghar' in surface or 'Asmari' in surface or 'Jahrum' in surface:
                zname = 'Ghar' if 'Ghar' in surface else ('Asmari-A' if 'Asmari_A' in surface else ('Asmari-B' if 'Asmari_B1' in surface else 'Jahrum'))
                md_top = wcfg['top'] + offset
                df_well.loc[df_well['DEPT'] >= md_top, 'ZONE'] = zname
                
        master_dfs.append(df_well)
        print(f"  ✓ Produced {clean_name}_synthetic.las ({n_rows:,} intervals)")

    # 2. Compile and distribute master datasets for ML Calibration and MBB
    master_df = pd.concat(master_dfs, ignore_index=True)
    master_df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
    master_df.to_csv(UCS_DATA_DIR / "master_dataset.csv", index=False)
    master_df.to_csv(D2S_DATA_DIR / "master_dataset_clean.csv", index=False)
    print("  ✓ Distributed Master Dataset CSV models to modules.")

    # 3. Generate and distribute Tops
    tops_path = OUTPUT_DIR / "Well_Tops_synthetic.txt"
    create_synthetic_tops(tops_path)
    shutil.copy2(tops_path, D2S_DATA_DIR / "Well_Tops_Cleaned.txt")
    print("  ✓ Produced Well_Tops_Cleaned.txt and shared globally.")

    # 4. Generate 2D grids for spatial and rock physics templates
    for prop in ['Zp', 'Zs', 'Porosity', 'Sw']:
        grid_file = RP_DATA_DIR / f"{prop}_2D.npy"
        np.random.seed(123)
        if prop == 'Zp':         grid = np.random.normal(12500, 1500, (201, 201))
        elif prop == 'Zs':       grid = np.random.normal(7200, 800, (201, 201))
        elif prop == 'Porosity': grid = np.random.normal(0.08, 0.03, (201, 201))
        else:                    grid = np.random.normal(0.35, 0.15, (201, 201))
        np.save(grid_file, grid)
    print("  ✓ Extracted and distributed 2D spatial numpy arrays.")

    print("\n" + "=" * 70)
    print("  Global Synthetic Data Generation Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
