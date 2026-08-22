"""
Multi-Model Subsidence Comparison Example
=========================================
Demonstrates how to run the Geertsma and DIF models on a synthetic 2D grid,
and compare them along with literature values (e.g., Segall) using the Comparator.
"""

import numpy as np
from modules.subsidence_analysis.src.influence_function import compute_dif_subsidence
from modules.subsidence_analysis.src.geertsma import calculate_pore_compressibility, calculate_geertsma_factor
from modules.subsidence_analysis.src.comparator import SubsidenceComparator

def main():
    print("=" * 70)
    print("  Subsidence Benchmarking: DIF vs Geertsma vs Others")
    print("=" * 70)

    # 1. Create a synthetic 10x10 km reservoir grid
    print("\n[1] Generating synthetic reservoir grid...")
    x = np.linspace(-5000, 5000, 50)
    y = np.linspace(-5000, 5000, 50)
    X, Y = np.meshgrid(x, y)
    
    # Mask a circular reservoir
    R_eq = 3000.0
    mask = (X**2 + Y**2) <= R_eq**2
    
    # Reservoir Properties
    depth_m = 2300.0
    h_eff_m = 100.0
    depletion_mpa = 15.0
    poisson = 0.25
    e_gpa = 27.0
    alpha = 0.825
    
    cm_val = calculate_pore_compressibility(e_gpa, poisson)
    
    # Compaction Grid (Cm * h * dP)
    compaction_grid = np.full(X.shape, np.nan)
    compaction_grid[mask] = alpha * cm_val * h_eff_m * depletion_mpa

    # 2. Run Geertsma Model
    print("[2] Running Geertsma (Analytical) Model...")
    geertsma_factor = calculate_geertsma_factor(depth_m, R_eq)
    max_geertsma_m = (alpha * cm_val * depletion_mpa * h_eff_m * geertsma_factor)
    max_geertsma_cm = max_geertsma_m * 100.0

    # 3. Run DIF (Nucleus of Strain) Model
    print("[3] Running DIF (Spatial Integration) Model...")
    # Using step=2 for faster execution in this example
    dif_grid = compute_dif_subsidence(
        compaction_grid_m=compaction_grid,
        x_grid_m=X, y_grid_m=Y,
        depth_m=depth_m, poisson_ratio=poisson, step=2
    )
    max_dif_cm = np.nanmax(dif_grid) * 100.0  # Convert to cm

    # 4. Compare using the SubsidenceComparator
    print("\n[4] Generating Master Comparison Table...")
    comparator = SubsidenceComparator(baseline_method="Geertsma")
    
    # Add computed models
    comparator.add_method_result("Geertsma", max_geertsma_cm)
    comparator.add_method_result("DIF (Nucleus of Strain)", max_dif_cm)
    
    # Add some hypothetical literature/other models (scaled to our depletion)
    comparator.add_scaled_literature_result("Segall (Literature)", 1.2, 5.0, depletion_mpa)
    comparator.add_method_result("Gibson (Double Drainage)", max_geertsma_cm * 0.85)

    # Print Report
    df_comparison = comparator.get_comparison_table()
    print("\n" + "-" * 70)
    print(df_comparison.to_string(index=False))
    print("-" * 70)
    
if __name__ == "__main__":
    main()
