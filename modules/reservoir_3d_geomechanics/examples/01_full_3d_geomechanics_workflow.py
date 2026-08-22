"""
End-to-End 3D Geomechanics Workflow Example
============================================
Runs the complete 3D modeling workflow using synthetic 3D cubes.
"""

import numpy as np
from pathlib import Path

from modules.reservoir_3d_geomechanics.data.generate_synthetic_3d_cubes import generate_synthetic_3d_grid
from modules.reservoir_3d_geomechanics.src import (
    compute_3d_dynamic_moduli,
    calibrate_dynamic_to_static_e,
    calibrate_e_to_ucs,
    compute_3d_static_properties,
    compute_3d_stress_field
)


def main():
    print("=" * 70)
    print("  3D RESERVOIR GEOMECHANICS WORKFLOW")
    print("=" * 70)

    # 1. Generate Synthetic Cubes
    data_dir = Path("./data/synthetic_3d")
    generate_synthetic_3d_grid(str(data_dir))

    # 2. Load 3D Cubes
    print("\n[1] Loading 3D seismic property arrays...")
    vp_3d = np.load(data_dir / "vp_3d.npy")
    vs_3d = np.load(data_dir / "vs_3d.npy")
    density_3d = np.load(data_dir / "density_3d.npy")
    depth_3d = np.load(data_dir / "depth_3d.npy")

    print(f"  Grid Dimensions: {vp_3d.shape} (Inline x Crossline x Depth)")

    # 3. Compute 3D Dynamic Properties
    print("\n[2] Computing 3D Dynamic Moduli...")
    dyn_props = compute_3d_dynamic_moduli(vp_3d, vs_3d, density_3d)
    print(f"  E_dyn mean: {np.mean(dyn_props['e_dyn_gpa']):.2f} GPa")
    print(f"  PR_dyn mean: {np.mean(dyn_props['pr_dyn']):.3f}")

    # 4. Calibrate with Well Data (Synthetic Well Arrays for Calibration)
    print("\n[3] Calibrating Dynamic-to-Static Properties...")
    fake_edyn_well = np.linspace(20, 60, 50)
    fake_estat_well = 0.65 * fake_edyn_well - 2.0
    fake_ucs_well = 2.8 * (fake_estat_well ** 0.85)

    e_calib = calibrate_dynamic_to_static_e(fake_edyn_well, fake_estat_well)
    ucs_calib = calibrate_e_to_ucs(fake_estat_well, fake_ucs_well)
    print(f"  E_stat Equation: {e_calib['eq']}")
    print(f"  UCS Equation:    {ucs_calib['eq']}")

    # 5. Compute 3D Static Properties
    print("\n[4] Generating 3D Static Moduli and UCS...")
    e_stat_3d, pr_stat_3d, ucs_3d, cm_3d = compute_3d_static_properties(
        dyn_props["e_dyn_gpa"],
        dyn_props["pr_dyn"],
        e_calib,
        ucs_calib,
        scale_factor=0.95
    )
    print(f"  E_static 3D mean: {np.mean(e_stat_3d):.2f} GPa")
    print(f"  UCS 3D mean:      {np.mean(ucs_3d):.2f} MPa")

    # 6. Compute 3D Stress Field
    print("\n[5] Calculating 3D Stress Tensor Field (Sv, Pp, Shmin, SHmax)...")
    stresses_3d = compute_3d_stress_field(
        density_3d,
        depth_3d,
        pr_stat_3d,
        pore_pressure_gradient_m_pa_m=0.0113,
        tectonic_stress_mpa=2.5,
        sh_sh_ratio=1.18
    )

    print(f"  Sv 3D mean:    {np.mean(stresses_3d['sv_mpa']):.2f} MPa")
    print(f"  Pp 3D mean:    {np.mean(stresses_3d['pp_mpa']):.2f} MPa")
    print(f"  Shmin 3D mean: {np.mean(stresses_3d['shmin_mpa']):.2f} MPa")
    print(f"  SHmax 3D mean: {np.mean(stresses_3d['shmax_mpa']):.2f} MPa")

    print("\n" + "=" * 70)
    print("  3D Geomechanical Modeling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
