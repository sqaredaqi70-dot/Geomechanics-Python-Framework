"""
3D Wellbore Stability & Mud Weight Window Demo
===============================================
Computes Mud Weight Window (MWW) for various well inclinations (0° to 90°)
and azimuths (0° to 360°) to generate a Stability Contour Map.
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.wellbore_stability_3d.src import calculate_mud_weight_window


def main():
    print("=" * 70)
    print("  3D WELLBORE STABILITY & MUD WEIGHT WINDOW ANALYSIS")
    print("=" * 70)

    # Reservoir Inputs
    depth_m = 2500.0
    sv_mpa = 55.0
    shmax_mpa = 60.0
    shmin_mpa = 45.0
    pp_mpa = 26.0
    ucs_mpa = 65.0
    shmax_azimuth = 35.0  # N35°E

    # Calculate for Vertical Well
    mww_vert = calculate_mud_weight_window(
        depth_m, sv_mpa, shmax_mpa, shmin_mpa, pp_mpa, ucs_mpa,
        shmax_azimuth_deg=shmax_azimuth, inclination_deg=0.0, azimuth_deg=0.0
    )

    print(f"\nVertical Well Stability Results (Depth = {depth_m:.0f} m):")
    print(f"  Pore Pressure:     {mww_vert['pore_pressure_sg']:.2f} g/cm³ (SG)")
    print(f"  Min MW (Breakout): {mww_vert['breakout_mw_sg']:.2f} g/cm³ (SG)")
    print(f"  Max MW (Tensile):  {mww_vert['tensile_mw_sg']:.2f} g/cm³ (SG)")
    print(f"  Overburden Stress: {mww_vert['overburden_sg']:.2f} g/cm³ (SG)")

    # Hemisphere Analysis for Deviated Wells (0-90° Inc, 0-360° Azi)
    print("\nComputing Stability Surface for all Trajectories (Stereonet)...")
    incs = np.linspace(0, 90, 31)
    azis = np.linspace(0, 360, 73)
    bo_map = np.zeros((len(incs), len(azis)))

    for i, inc in enumerate(incs):
        for j, azi in enumerate(azis):
            res = calculate_mud_weight_window(
                depth_m, sv_mpa, shmax_mpa, shmin_mpa, pp_mpa, ucs_mpa,
                shmax_azimuth_deg=shmax_azimuth, inclination_deg=inc, azimuth_deg=azi
            )
            bo_map[i, j] = res["breakout_mw_sg"]

    # Plot Contour Map
    AZI, INC = np.meshgrid(azis, incs)
    fig, ax = plt.subplots(figsize=(10, 8))
    cp = ax.contourf(AZI, INC, bo_map, levels=20, cmap="YlOrRd")
    cbar = plt.colorbar(cp, ax=ax)
    cbar.set_label("Required Breakout Mud Weight (g/cm³ SG)")

    ax.set_xlabel("Well Azimuth (deg)")
    ax.set_ylabel("Well Inclination (deg)")
    ax.set_title("3D Wellbore Stability Map (Breakout Risk)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("./mww_stability_map.png", dpi=300)
    print("✅ Stability Contour Map saved to './mww_stability_map.png'")


if __name__ == "__main__":
    main()
