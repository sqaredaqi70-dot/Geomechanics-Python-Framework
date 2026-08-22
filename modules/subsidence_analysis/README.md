# 🌍 Subsidence Analysis Module

Analytical modeling of production-induced surface subsidence based on **Geertsma (1973)** poroelastic theory.

## 📋 Features
- Pore volume compressibility calculation from elastic properties
- Geertsma geometry factor for disk-shaped reservoirs
- Maximum surface subsidence at reservoir center
- Radial subsidence profile
- Fault reactivation risk assessment (Ts_norm)

## 🚀 Quick Start

```python
from modules.subsidence_analysis.src import calculate_max_subsidence

# Example: Calculate subsidence for 10 MPa depletion
subsidence_cm = calculate_max_subsidence(
    depletion_mpa=10.0,
    h_eff_m=107.0,
    depth_m=2275.0,
    radius_m=5000.0,
    e_modulus_gpa=27.62,
    poisson_ratio=0.231,
    alpha_biot=0.825
)
print(f"Max subsidence: {subsidence_cm:.3f} cm")
📊 Generate Synthetic Data
Bash

python modules/subsidence_analysis/data/generate_synthetic_grid.py
📚 Reference
Geertsma, J. (1973). "Land subsidence above compacting oil and gas reservoirs."
Journal of Petroleum Technology, 25(06), 734-744.

⚖️ License
Dual License (AGPL-3.0 for academic / Commercial license for industrial use)
