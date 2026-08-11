"""
═══════════════════════════════════════════════════════════════════════
                  SYNTHETIC WELL DATA GENERATOR
═══════════════════════════════════════════════════════════════════════

Generates realistic synthetic LAS files and formation tops for
demonstration of the Geomechanics Python Framework.

⚠ IMPORTANT NOTICE:
   All generated data is 100% SYNTHETIC and computer-generated.
   It does NOT represent any real oil/gas field, well, or reservoir.
   Depths, log values, and formation tops are arbitrary.
   Use ONLY for testing and educational purposes.

Author:   Sqared Aqi
Contact:  sqaredaqi70@gmail.com
License:  AGPL-3.0 (Academic) / Commercial (see LICENSE files)
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import lasio
except ImportError:
    print("ERROR: 'lasio' package not found.")
    print("Install with: pip install lasio")
    exit(1)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR = Path(__file__).parent / "data" / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generic well definitions — no real field information
# Depths (in meters) are arbitrary and for demonstration only
WELLS = {
    'Well_A': {'top': 2900.0, 'bot': 3550.0, 'seed': 42},
    'Well_B': {'top': 2950.0, 'bot': 3600.0, 'seed': 123},
    'Well_C': {'top': 3050.0, 'bot': 3700.0, 'seed': 456},
}

# Synthetic formation tops (offsets from well top)
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


# ══════════════════════════════════════════════════════════════
# SYNTHETIC LOG GENERATORS
# ══════════════════════════════════════════════════════════════
def generate_gr(depth, seed=42):
    """Gamma ray log (API) — carbonate/shale variation."""
    rng = np.random.default_rng(seed)
    base = 25 + 15 * np.sin(depth * 0.003)
    noise = rng.normal(0, 8, len(depth))
    shale_events = rng.random(len(depth)) < 0.05
    gr = base + noise + shale_events * rng.uniform(40, 80, len(depth))
    return np.clip(gr, 10, 200)


def generate_density(depth, seed=42):
    """Bulk density (g/cc) — carbonate with compaction trend."""
    rng = np.random.default_rng(seed)
    compaction = 2.35 + 0.00015 * depth
    noise = rng.normal(0, 0.05, len(depth))
    porosity_effect = -0.15 * np.sin(depth * 0.005)
    rho = compaction + noise + porosity_effect
    return np.clip(rho, 2.0, 2.85)


def generate_sonic_p(depth, seed=42):
    """Compressional sonic DT (us/ft)."""
    rng = np.random.default_rng(seed)
    base = 65 - 0.003 * depth
    noise = rng.normal(0, 4, len(depth))
    dt = base + noise + 8 * np.sin(depth * 0.004)
    return np.clip(dt, 45, 110)


def generate_sonic_s(dt_p, seed=42):
    """Shear sonic DTS (us/ft) — from Vp/Vs ratio."""
    rng = np.random.default_rng(seed)
    vp_vs_ratio = 1.85
    noise = rng.normal(0, 3, len(dt_p))
    return dt_p * vp_vs_ratio + noise


def generate_ucs(depth, gr, seed=42):
    """UCS in kPa (workflow expects kPa)."""
    rng = np.random.default_rng(seed)
    ucs_base_mpa = 50 + 30 * np.sin(depth * 0.003)
    shale_penalty = np.where(gr > 60, -20, 0)
    ucs_mpa = np.clip(
        ucs_base_mpa + shale_penalty + rng.normal(0, 10, len(depth)),
        5, 150
    )
    return ucs_mpa * 1000  # MPa → kPa


def generate_stresses(depth, seed=42):
    """Sv, SHmax, Shmin in kPa (strike-slip regime)."""
    rng = np.random.default_rng(seed)
    # Vertical stress (~22.6 kPa/m ≈ 1 psi/ft)
    sv_mpa = 22.6 * depth / 1000
    # Strike-slip regime: SHmax > Sv > Shmin
    sHmax_mpa = sv_mpa * (1.05 + rng.normal(0, 0.05, len(depth)))
    shmin_mpa = sv_mpa * (0.82 + rng.normal(0, 0.04, len(depth)))
    return sv_mpa * 1000, sHmax_mpa * 1000, shmin_mpa * 1000


def generate_pore_pressure(depth, seed=42):
    """Pore pressure (kPa) — near-hydrostatic."""
    rng = np.random.default_rng(seed)
    pp_mpa = 10.5 * depth / 1000 + rng.normal(0, 0.5, len(depth))
    return pp_mpa * 1000


def generate_moduli(depth, dt_p, seed=42):
    """Young's modulus (GPa) and Poisson's ratio."""
    rng = np.random.default_rng(seed)
    e_dyn_gpa = 50 * (65 / dt_p) ** 2 + rng.normal(0, 5, len(depth))
    e_dyn_gpa = np.clip(e_dyn_gpa, 10, 90)
    e_sta_gpa = 0.7 * e_dyn_gpa + rng.normal(0, 3, len(depth))
    pr_dyn = 0.28 + 0.05 * np.sin(depth * 0.002) + \
             rng.normal(0, 0.02, len(depth))
    pr_sta = pr_dyn * 0.9
    return (e_dyn_gpa, np.clip(e_sta_gpa, 5, 70),
            np.clip(pr_dyn, 0.15, 0.40), np.clip(pr_sta, 0.15, 0.40))


def generate_friction_cohesion(gr, ucs_kpa, seed=42):
    """Friction angle (deg) and cohesion (kPa)."""
    rng = np.random.default_rng(seed)
    fang = 35 - 0.15 * gr + rng.normal(0, 3, len(gr))
    fang = np.clip(fang, 15, 45)
    cohesion_kpa = ucs_kpa * 0.15 + rng.normal(0, 500, len(gr))
    return fang, np.clip(cohesion_kpa, 100, 20000)


# ══════════════════════════════════════════════════════════════
# LAS FILE WRITER
# ══════════════════════════════════════════════════════════════
def create_synthetic_las(well_name, well_config, out_path):
    """Create a synthetic LAS file for one well."""
    seed = well_config['seed']
    top = well_config['top']
    bot = well_config['bot']

    # Depth array — 0.1524 m sampling (= 0.5 ft, typical)
    depth = np.arange(top, bot + 0.1524, 0.1524)
    n = len(depth)

    # Generate all logs
    gr = generate_gr(depth, seed)
    rho = generate_density(depth, seed)
    dt_p = generate_sonic_p(depth, seed)
    dt_s = generate_sonic_s(dt_p, seed)
    ucs_kpa = generate_ucs(depth, gr, seed)
    sv_kpa, sHmax_kpa, shmin_kpa = generate_stresses(depth, seed)
    pp_kpa = generate_pore_pressure(depth, seed)
    e_dyn, e_sta, pr_dyn, pr_sta = generate_moduli(depth, dt_p, seed)
    fang, cohesion_kpa = generate_friction_cohesion(gr, ucs_kpa, seed)

    # Tensile strength (approximate: 10% of UCS)
    tstr_kpa = ucs_kpa * 0.10

    # Assemble DataFrame with all expected curves
    df = pd.DataFrame({
        'DEPT':          depth,
        'GR':            gr,
        'DT':            dt_p,
        'DTS':           dt_s,
        'RHOB':          rho,
        # UCS variants
        'UCS_FINAL':     ucs_kpa,
        'UCS_FINAL_70':  ucs_kpa * 0.70,
        'UCS_FINAL_80':  ucs_kpa * 0.80,
        'UCS_FINAL_90':  ucs_kpa * 0.90,
        'UCS_FINAL_110': ucs_kpa * 1.10,
        'UCS_FINAL_120': ucs_kpa * 1.20,
        'UCS_HORSRUD':   ucs_kpa * (1 + np.random.normal(0, 0.10, n)),
        'UCS_MCNALLY':   ucs_kpa * (1 + np.random.normal(0, 0.12, n)),
        'UCS_CDE':       ucs_kpa * (1 + np.random.normal(0, 0.15, n)),
        'UCS_SMG_RPC':   ucs_kpa * (1 + np.random.normal(0, 0.13, n)),
        'UCS_SND_RPC':   ucs_kpa * (1 + np.random.normal(0, 0.14, n)),
        'UCS_YME':       ucs_kpa * (1 + np.random.normal(0, 0.11, n)),
        # Tensile strength
        'TSTR_10%':      tstr_kpa,
        'TSTR_8_4%':     tstr_kpa * 0.84,
        # Stresses
        'SHMAX_PHS':     sHmax_kpa,
        'SHMAX_MC_UB':   sHmax_kpa * 1.05,
        'SHMIN_PHS':     shmin_kpa,
        'SHMIN_MC_LB':   shmin_kpa * 0.95,
        # Pore pressure
        'FINAL_PP':      pp_kpa,
        'FINAL_PP_COR':  pp_kpa * 1.02,
        # Elastic moduli
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
        # Rock strength
        'FANG_FROMGR':         fang,
        'COHESION_FROM_GR':    cohesion_kpa,
        'COHESION_FROM_PLUMB': cohesion_kpa * 0.95,
    })

    # Build LAS file
    las = lasio.LASFile()
    las.well['WELL'] = well_name
    las.well['COMP'] = 'Synthetic Data - Educational Use'
    las.well['FLD']  = 'Synthetic Field'
    las.well['LOC']  = 'N/A - Synthetic'
    las.well['CTRY'] = 'N/A'
    las.well['SRVC'] = 'Geomechanics Python Framework - Synthetic Generator v1.0'
    las.well['DATE'] = '2024-01-01'
    las.well['UWI']  = f'SYN-{well_name}'

    las.well['STRT'].value = float(depth[0])
    las.well['STOP'].value = float(depth[-1])
    las.well['STEP'].value = 0.1524
    las.well['NULL'].value = -999.25

    # Add all curves
    for col in df.columns:
        unit = 'M' if col == 'DEPT' else ''
        las.add_curve(col, df[col].values, unit=unit,
                       descr=f'Synthetic {col}')

    las.write(str(out_path))
    return len(df)


# ══════════════════════════════════════════════════════════════
# WELL TOPS WRITER
# ══════════════════════════════════════════════════════════════
def create_synthetic_tops(out_path):
    """Create synthetic well tops file."""
    lines = [
        "# ══════════════════════════════════════════════════════════════",
        "#           SYNTHETIC WELL TOPS - EDUCATIONAL USE ONLY",
        "# ══════════════════════════════════════════════════════════════",
        "#",
        "# ⚠ All depths in this file are SYNTHETIC and computer-generated.",
        "#   They do NOT represent any real oil/gas field or well.",
        "#",
        "# Generated by: Geomechanics Python Framework",
        "# Author:       Sqared Aqi (sqaredaqi70@gmail.com)",
        "# ══════════════════════════════════════════════════════════════",
        "",
        f"{'ID':>4} {'Well':10} {'Surface':15} {'MD':>10} {'Z':>10}",
        "=" * 55,
    ]

    idx = 1
    for well_name, wcfg in WELLS.items():
        for surface, offset in ZONE_OFFSETS.items():
            md = wcfg['top'] + offset
            z = -md  # TVDSS (negative, below sea level)
            lines.append(
                f"{idx:>4} {well_name:10} {surface:15} "
                f"{md:>10.2f} {z:>10.2f}"
            )
            idx += 1

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
def main():
    """Generate all synthetic data files."""
    print("=" * 60)
    print("  Synthetic Well Data Generator")
    print("  Geomechanics Python Framework")
    print("  ⚠ For educational and demonstration purposes only")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # Generate LAS files
    print("Generating synthetic LAS files:")
    for well_name, wcfg in WELLS.items():
        out_path = OUTPUT_DIR / f"{well_name}_synthetic.las"
        n_rows = create_synthetic_las(well_name, wcfg, out_path)
        print(f"  ✅ {out_path.name}  ({n_rows:,} rows)")

    # Generate tops file
    tops_path = OUTPUT_DIR / "Well_Tops_synthetic.txt"
    create_synthetic_tops(tops_path)
    print(f"  ✅ {tops_path.name}")

    # Generate README
    readme_content = """# Synthetic Sample Data

## ⚠ Important Notice

All data in this folder is **synthetic and computer-generated**
for demonstration purposes.

- ❌ Does **NOT** represent any real oil/gas field
- ❌ Depths and log values are **arbitrary**
- ✅ Suitable for testing the workflow logic

## Files

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `Well_A_synthetic.las`      | Synthetic wireline logs for Well A   |
| `Well_B_synthetic.las`      | Synthetic wireline logs for Well B   |
| `Well_C_synthetic.las`      | Synthetic wireline logs for Well C   |
| `Well_Tops_synthetic.txt`   | Formation tops for all wells         |

## Regenerating

To regenerate this data:

```bash
python generate_synthetic_data.py
  Real Data
Real field data cannot be shared due to confidentiality agreements.
This synthetic data allows testing of the workflow methodology.
"""
readme_path = OUTPUT_DIR / "README.md"
with open(readme_path, 'w', encoding='utf-8') as f:
f.write(readme_content)
print(f" ✅ {readme_path.name}")
print("\n" + "=" * 60)
print(f"  ✅ All files generated in: {OUTPUT_DIR}")
print("=" * 60)
if name == 'main':
main()
