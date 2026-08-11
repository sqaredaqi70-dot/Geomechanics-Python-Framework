                        MAIN WORKFLOW

Main entry point for the 1D Mechanical Earth Model workflow.

Usage
-----
    python -m modules.geomechanics_1d_mem.src.main

    OR (from project root):

    python modules/geomechanics_1d_mem/src/main.py

Author:  Saeed Gharedaghi
Contact: sqaredaqi70@gmail.com
License: AGPL-3.0 (Academic) / Commercial (see LICENSE files)
═══════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Add project root to Python path (for direct execution)
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

from modules.geomechanics_1d_mem.src import config as cfg
from modules.geomechanics_1d_mem.src import data_loader
from modules.geomechanics_1d_mem.src import geomechanics
from modules.geomechanics_1d_mem.src import visualization


# MAIN WORKFLOW
def main():
    """Execute the complete 1D MEM workflow."""
    print("=" * 70)
    print("  Geomechanics Python Framework")
    print("  Module: 1D Mechanical Earth Model")
    print("  Author: Sqared Aqi")
    print("=" * 70)

    # ─── Step 1: Load data ───────────────────────────────────
    print("\n[1/4] Loading well data...")
    wells = data_loader.load_all_wells()

    if len(wells) == 0:
        print("\n❌ ERROR: No wells loaded!")
        print("   Please run 'python generate_synthetic_data.py' first.")
        return

    print(f"\n  ✅ Loaded {len(wells)} wells")

    # ─── Step 2: Process each well ────────────────────────────
    print("\n[2/4] Processing wells (computing MEM)...")
    for name in list(wells.keys()):
        wells[name] = geomechanics.process_well(wells[name])
        print(f"  ✅ {name}: MEM computed")

    # ─── Step 3: Determine overall regime ─────────────────────
    print("\n[3/4] Determining stress regime...")
    regime = geomechanics.determine_overall_regime(wells)
    print(f"  Regime: {regime['regime_str']}")
    print(f"  K₀ = {regime['K0']:.3f}")
    print(f"  kH = {regime['kH']:.3f}")

    # ─── Step 4: Generate visualizations ─────────────────────
    print("\n[4/4] Generating figures...")
    figures_dir = cfg.OUTPUT_DIR / "figures"

    # Well location map
    visualization.plot_well_locations(wells, figures_dir)

    # Per-well MEM profiles
    for name, wd in wells.items():
        visualization.plot_mem_profile(wd, figures_dir)

    # Stress polygon
    visualization.plot_stress_polygon(wells, regime, figures_dir)

    # ─── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {cfg.OUTPUT_DIR}")
    print(f"  Figures generated in: {figures_dir}\n")


# ENTRY POINT
if __name__ == '__main__':
    main()
