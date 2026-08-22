"""
Comprehensive Visualization Example
====================================
Demonstrates the full visualization capabilities of the Subsidence Analysis
Module: 1D profiles, 2D maps, 3D surfaces, MC distributions, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt

from modules.subsidence_analysis.src import (
    apply_publication_style,
    # Models
    calculate_subsidence_profile,
    MCConfig, MonteCarloSubsidence, tornado_analysis, sobol_analysis,
    # Visualization
    plot_radial_profile,
    plot_subsidence_vs_depletion,
    plot_subsidence_map,
    plot_risk_zonation_map,
    plot_3d_deformed_surface,
    plot_mc_distributions,
    plot_tornado_sensitivity,
    plot_sobol_indices,
    save_figure,
)


def main():
    print("=" * 70)
    print("  Subsidence Visualization Showcase")
    print("=" * 70)

    apply_publication_style()

    # ─── Common Parameters ─────────────────────────────────────
    h_eff = 107.0
    depth = 2275.0
    radius = 5000.0
    e_gpa = 27.62
    poisson = 0.231
    depletion = 10.0

    # ─── 1. Radial profile ────────────────────────────────────
    print("\n[1] Plotting radial profile...")
    r_km = np.linspace(0, 15, 200)
    sub_cm = calculate_subsidence_profile(
        r_distance_m=r_km * 1000,
        depletion_mpa=depletion,
        h_eff_m=h_eff, depth_m=depth, radius_m=radius,
        e_modulus_gpa=e_gpa, poisson_ratio=poisson
    )
    fig1 = plot_radial_profile(r_km, sub_cm,
                                title=f"Radial Profile — ΔPp = {depletion} MPa")
    save_figure(fig1, "01_radial_profile", "./example_outputs")

    # ─── 2. Subsidence vs. Depletion ──────────────────────────
    print("[2] Plotting subsidence vs. depletion...")
    dpp_range = np.linspace(0, 30, 100)
    from modules.subsidence_analysis.src import calculate_max_subsidence
    sub_range = calculate_max_subsidence(
        depletion_mpa=dpp_range,
        h_eff_m=h_eff, depth_m=depth, radius_m=radius,
        e_modulus_gpa=e_gpa, poisson_ratio=poisson
    )
    fig2 = plot_subsidence_vs_depletion(dpp_range, sub_range)
    save_figure(fig2, "02_subsidence_vs_depletion", "./example_outputs")

    # ─── 3. Synthetic 2D subsidence map ───────────────────────
    print("[3] Plotting 2D subsidence map...")
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    R_km = np.sqrt(X**2 + Y**2)
    R_m = R_km * 1000

    sub_map = calculate_subsidence_profile(
        r_distance_m=R_m, depletion_mpa=depletion,
        h_eff_m=h_eff, depth_m=depth, radius_m=radius,
        e_modulus_gpa=e_gpa, poisson_ratio=poisson
    )
    sub_map[R_km > 12] = np.nan  # mask outside domain

    wells = {"Well-A": (0.0, 0.0), "Well-B": (1.7, 0.3), "Well-C": (-1.2, -2.8)}
    fig3 = plot_subsidence_map(X, Y, sub_map,
                                title="2D Subsidence Map — Synthetic Reservoir",
                                well_coords_km=wells)
    save_figure(fig3, "03_subsidence_map_2d", "./example_outputs")

    # ─── 4. Risk zonation ─────────────────────────────────────
    print("[4] Plotting risk zonation map...")
    fig4 = plot_risk_zonation_map(X, Y, sub_map * 3, well_coords_km=wells)
    save_figure(fig4, "04_risk_zonation", "./example_outputs")

    # ─── 5. 3D deformed surface ───────────────────────────────
    print("[5] Plotting 3D deformed reservoir surface...")
    Z_reservoir = 2275 - 300 * np.exp(-(R_km / 5)**2)
    Z_reservoir[R_km > 12] = np.nan
    fig5 = plot_3d_deformed_surface(X, Y, Z_reservoir, sub_map,
                                     title="3D Deformed Surface — Synthetic Anticline")
    save_figure(fig5, "05_3d_deformed_surface", "./example_outputs")

    # ─── 6. Monte Carlo Distributions ─────────────────────────
    print("[6] Running Monte Carlo & plotting distributions...")
    config = MCConfig(n_iterations=5000, random_seed=42)
    engine = MonteCarloSubsidence(config)
    results = engine.run_all_scenarios([5, 10, 15, 20])
    fig6 = plot_mc_distributions(results,
                                  title="MC Subsidence Distributions — 4 Scenarios")
    save_figure(fig6, "06_mc_distributions", "./example_outputs")

    # ─── 7. Tornado Sensitivity ───────────────────────────────
    print("[7] Plotting Tornado sensitivity...")
    tornado_df = tornado_analysis(config, depletion_mpa=10.0)
    base_val = results[10]["mean"]
    fig7 = plot_tornado_sensitivity(tornado_df, baseline=base_val)
    save_figure(fig7, "07_tornado_sensitivity", "./example_outputs")

    # ─── 8. Sobol Indices ─────────────────────────────────────
    print("[8] Plotting Sobol indices...")
    sobol_df = sobol_analysis(config, depletion_mpa=10.0, n_samples=1024)
    fig8 = plot_sobol_indices(sobol_df)
    save_figure(fig8, "08_sobol_indices", "./example_outputs")

    print("\n" + "=" * 70)
    print("  All figures saved to ./example_outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
