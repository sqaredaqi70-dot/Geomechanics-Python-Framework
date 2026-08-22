"""
Monte Carlo Subsidence Analysis - Example Usage
================================================
Demonstrates probabilistic subsidence assessment with sensitivity analysis
for a generic carbonate reservoir.
"""

import numpy as np
from modules.subsidence_analysis.src.monte_carlo import (
    MCConfig, MonteCarloSubsidence,
    tornado_analysis, sobol_analysis, time_evolution
)


def main():
    print("=" * 70)
    print("  Monte Carlo Subsidence Analysis - Example")
    print("  Reservoir: Generic Carbonate (synthetic data)")
    print("=" * 70)
    
    # ────────────────────────────────────────
    # 1. Configure the Monte Carlo simulation
    # ────────────────────────────────────────
    config = MCConfig(n_iterations=10_000, random_seed=42)
    
    print(f"\nMC Configuration:")
    print(f"  E_static:  {config.E_static.mean:.2f} ± {config.E_static.std:.2f} GPa")
    print(f"  Poisson:   {config.poisson_ratio.mean:.3f} ± {config.poisson_ratio.std:.3f}")
    print(f"  NTG:       {config.ntg.mean:.3f} ± {config.ntg.std:.3f}")
    print(f"  h_gross:   {config.h_gross.mean:.1f} ± {config.h_gross.std:.1f} m")
    print(f"  Iterations: {config.n_iterations:,}")
    
    # ────────────────────────────────────────
    # 2. Run Monte Carlo scenarios
    # ────────────────────────────────────────
    engine = MonteCarloSubsidence(config)
    scenarios = [5, 10, 15, 20, 30]
    results = engine.run_all_scenarios(scenarios)
    
    print(f"\n{'-'*70}")
    print("MC Results Summary:")
    print(f"{'-'*70}")
    print(engine.get_summary_dataframe().to_string(index=False))
    
    # ────────────────────────────────────────
    # 3. Tornado sensitivity analysis
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Tornado Sensitivity (ΔPp = 10 MPa):")
    print(f"{'-'*70}")
    tornado_df = tornado_analysis(config, depletion_mpa=10.0)
    print(tornado_df.to_string(index=False))
    
    # ────────────────────────────────────────
    # 4. Sobol variance-based sensitivity
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Sobol Sensitivity Indices (ΔPp = 10 MPa):")
    print(f"{'-'*70}")
    sobol_df = sobol_analysis(config, depletion_mpa=10.0, n_samples=2048)
    print(sobol_df.to_string(index=False))
    
    # ────────────────────────────────────────
    # 5. Time evolution (2 MPa/year over 10 years)
    # ────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Time Evolution (2 MPa/year):")
    print(f"{'-'*70}")
    years = np.arange(0, 11)
    time_df = time_evolution(engine, years, depletion_rate_mpa_per_year=2.0)
    print(time_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("  Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
