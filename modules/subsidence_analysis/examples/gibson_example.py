"""
Gibson/Terzaghi Consolidation - Example Usage
==============================================
Demonstrates time-dependent subsidence with coupled Geertsma-Gibson
framework for a generic carbonate reservoir under production-induced depletion.
"""

import numpy as np
import pandas as pd
from modules.subsidence_analysis.src.gibson import (
    GibsonSubsidence, GibsonScenario, DEFAULT_SCENARIOS,
    terzaghi_degree_of_consolidation, compute_time_factor
)


def generate_synthetic_depletion_history(
    total_years: float = 50.0,
    peak_depletion_mpa: float = 15.0,
    n_points: int = 500
) -> pd.DataFrame:
    """
    Generate a realistic synthetic depletion history mimicking Eclipse output.
    Uses a smooth exponential build-up + decline profile.
    """
    t_years = np.linspace(0, total_years, n_points)
    t_seconds = t_years * 365.25 * 86400
    # Sinusoidal build-up mimicking reservoir depletion cycle
    depletion_mpa = peak_depletion_mpa * np.sin(np.pi * t_years / total_years)
    depletion_mpa = np.clip(depletion_mpa, 0, None)
    return pd.DataFrame({
        "year": t_years,
        "time_seconds": t_seconds,
        "depletion_mpa": depletion_mpa,
    })


def main():
    print("=" * 70)
    print("  Gibson/Terzaghi Consolidation Analysis - Example")
    print("=" * 70)

    # ────────────────────────────────────────
    # 1. Setup reservoir properties
    # ────────────────────────────────────────
    calculator = GibsonSubsidence(
        h_eff_m=107.0,        # Effective thickness
        depth_m=2275.0,       # Reservoir depth
        radius_m=5000.0,      # Equivalent radius
        e_modulus_gpa=27.62,  # Young's modulus
        poisson_ratio=0.231,  # Poisson's ratio
        alpha_biot=0.825      # Biot coefficient
    )

    print(f"\nReservoir Setup:")
    print(f"  Pore compressibility (Cm): {calculator.cm:.4e} 1/MPa")
    print(f"  Geertsma factor:           {calculator.geertsma_factor:.4f}")

    # ────────────────────────────────────────
    # 2. Generate synthetic depletion history
    # ────────────────────────────────────────
    df_history = generate_synthetic_depletion_history(
        total_years=50.0,
        peak_depletion_mpa=15.0
    )
    print(f"\nDepletion history:")
    print(f"  Duration:     {df_history['year'].max():.0f} years")
    print(f"  Peak depletion: {df_history['depletion_mpa'].max():.2f} MPa")

    # ────────────────────────────────────────
    # 3. Run all default consolidation scenarios
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Running Gibson Scenarios:")
    print(f"{'-'*70}")
    for sc in DEFAULT_SCENARIOS:
        print(f"  • {sc.name}: Cv={sc.cv_m2_per_s:.0e} m²/s, drainage={sc.drainage}")

    time_series, summary = calculator.run_all_scenarios(df_history)

    print(f"\n{'-'*70}")
    print("Summary Statistics:")
    print(f"{'-'*70}")
    print(summary.to_string(index=False))

    # ────────────────────────────────────────
    # 4. Instantaneous check
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Instantaneous Geertsma vs Time-Dependent Gibson (at t=10 years):")
    print(f"{'-'*70}")
    idx = np.argmin(np.abs(time_series["year"].values - 10.0))
    row = time_series.iloc[idx]
    print(f"  Depletion at t=10 yr: {row['depletion_mpa']:.2f} MPa")
    for sc in DEFAULT_SCENARIOS:
        u = row[f"{sc.name}_U"]
        gib = row[f"{sc.name}_gibson_cm"]
        geo = row[f"{sc.name}_geertsma_cm"]
        print(f"  {sc.name:22s}: U={u:.3f} | Gibson={gib:.4f} cm | Geertsma={geo:.4f} cm")

    # ────────────────────────────────────────
    # 5. Custom scenario example
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Custom User-Defined Scenario:")
    print(f"{'-'*70}")
    custom = GibsonScenario(
        name="TIGHT_CARBONATE",
        cv_m2_per_s=5e-7,
        drainage="single",
        description="User-specified tight carbonate case"
    )
    ts_custom, sum_custom = calculator.run_all_scenarios(
        df_history, scenarios=[custom]
    )
    print(sum_custom.to_string(index=False))

    print(f"\n{'='*70}")
    print("  Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
