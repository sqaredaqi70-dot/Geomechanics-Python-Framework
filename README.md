# 🌋 Geomechanics Python Framework

A Comprehensive Python Framework for Petroleum Geomechanics, 1D/3D Mechanical Earth Modeling, Rock Physics, Fault Reactivation Analysis, and Carbonate Reservoir Characterization.

![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Commercial License](https://img.shields.io/badge/License-Commercial_Available-gold.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![Framework Version](https://img.shields.io/badge/Version-v3.0.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Production_Stable-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows_%7C_Linux_%7C_macOS-lightgrey.svg)

---

## 📖 Overview

**Geomechanics Python Framework** is an open-source, modular, and extensible software toolkit designed for subsurface engineering, petroleum geomechanics, and reservoir management. It integrates well-log processing, 1D MEM construction, 3D seismic-driven geomechanical modeling, fault reactivation risk assessment, rock physics fluid substitution, and machine-learning-based carbonate characterization into unified, reproducible workflows.

The computational engines implement methodologies consistent with peer-reviewed research in petroleum geomechanics and geophysics (manuscripts currently under review in Q1 journals). Although tailored and calibrated for **heterogeneous carbonate reservoirs**, the framework is fully parameterizable and adaptable to any geological setting worldwide (sandstones, shales, chalks, tight formations).

### Core Capabilities at a Glance

- **Deterministic & Probabilistic Geomechanics:** 1D/3D stress modeling with 10,000-sample Monte Carlo uncertainty propagation and global Sobol sensitivity decomposition.
- **Fault Reactivation & Injection Safety:** 3D traction resolution on arbitrary fault planes, slip/dilation tendency analysis, and safe injection pressure limit ($\Delta P_p$) quantification.
- **Rock Physics & AVO:** Lambda-Rho / Mu-Rho fluid discrimination, Gassmann fluid substitution, and Voigt-Reuss-Hill velocity-porosity bounds.
- **Carbonate UCS Calibration:** Benchmarking of 11 published dynamic-to-static correlations, zone-specific power-law fitting, and ensemble ML (RF / GB / XGB) with honest Leave-One-Well-Out validation.
- **Spatial Autocorrelation Correction:** Moving Block Bootstrap (MBB) for correcting confidence intervals in spatially correlated wireline log data.

---

## 📦 Included Modules & Scientific Roadmap

| # | Module | Core Methodology & Features | Status | Version |
|---|--------|---------------------------|:------:|:-------:|
| 1 | 🏔️ **1D MEM** | 1D Mechanical Earth Modeling — multi-method UCS (Horsrud, McNally, CDE), elastic moduli, overburden & in-situ stress ($S_v$, $S_{Hmax}$, $S_{hmin}$, $P_p$), Mohr-Coulomb failure | ✅ Complete | `v1.0` |
| 2 | 🌍 **Subsidence Analysis** | Geertsma (1973) poroelastic disc, Gibson/Terzaghi 1D consolidation, Nucleus of Strain DIF kernel, Monte Carlo + Sobol/Tornado sensitivity | ✅ Complete | `v2.0` |
| 3 | 🎯 **3D Reservoir Geomechanics** | Seismic-driven 3D properties ($E$, $\nu$, UCS) with dynamic-to-static well calibration, 3D stress tensor ($S_v$, $P_p$, $S_{hmin}$, $S_{Hmax}$) | ✅ Complete | `v2.5` |
| 4 | ⚡ **Fault Reactivation** | 3D traction resolution ($T_s$, $T_d$, CFF), Andersonian taxonomy, 10k Monte Carlo PoF, global Sobol indices ($S_1$, $S_T$), safe $\Delta P_p$ limits | ✅ Complete | `v3.0` |
| 5 | 🔬 **Rock Physics** | LMR ($\lambda\rho$–$\mu\rho$) discrimination, Gassmann fluid substitution (brine/oil/gas), VRH bounds, 3D fluid contact surface mapping | ✅ Complete | `v3.0` |
| 6 | 🧪 **UCS Calibration & ML** | 11 published correlation benchmark, zone-specific power-laws, ensemble ML (RF, GB, XGB) with honest LOWO cross-validation | ✅ Complete | `v3.0` |
| 7 | 📊 **Dynamic-to-Static** | Pseudo-static linear scaling, Leave-One-Well-Out transfer test, Moving Block Bootstrap (MBB) autocorrelation correction, $N_{eff}$ estimation | ✅ Complete | `v3.0` |
| 8 | 🔧 **Wellbore Stability 3D** | Deviated trajectory optimization, Kirsch cylinder solutions, mud weight window, shear breakout / tensile fracture zonation | 📋 Planned | `v3.5` |

---

## ✨ Key Features

### 1D Earth Modeling
- Multi-method UCS estimation (Horsrud, McNally, CDE, SMG-RPC, SND-RPC, YME-based)
- Dynamic-to-static elastic moduli conversion with core-calibrated power-laws
- Overburden stress integration and pore pressure gradient computation
- Stress regime classification (Normal / Strike-Slip / Reverse via Zoback polygon)

### Advanced Subsidence Engine
- **Geertsma (1973):** Analytical poroelastic solution for disk-shaped reservoirs
- **Monte Carlo Uncertainty:** 10,000+ realizations with truncated distributions, Tornado sensitivity, and Sobol variance-based indices ($S_1$, $S_T$)
- **Gibson / Terzaghi Consolidation:** Time-dependent subsidence coupled with 1D consolidation theory and $C_v$ sensitivity
- **Depth Influence Function (DIF):** 2D spatial convolution via Nucleus of Strain kernel
- **Multi-Model Comparator:** Cross-method benchmarking and validation

### 3D Fault Reactivation & Injection Safety
- Full 3D traction resolution on arbitrarily oriented fault planes from seismic point clouds
- Andersonian kinematic classification (Strike-Slip / Normal / Reverse taxonomy)
- Normalized slip tendency ($T_{s,norm}$), dilation tendency ($T_d$), and Coulomb Failure Function (CFF)
- Probabilistic Probability of Failure (PoF) via 10,000-sample Monte Carlo simulation
- Global Sobol variance decomposition for identifying dominant geomechanical drivers
- Deterministic and probabilistic safe injection pressure limits ($\Delta P_p$)

### Rock Physics & Fluid Substitution
- Lambda-Rho vs. Mu-Rho ($\lambda\rho$–$\mu\rho$) litho-fluid discrimination templates
- Gassmann fluid substitution engine (brine → oil → gas scenarios)
- Voigt-Reuss-Hill theoretical velocity-porosity bounds
- 3D structural fluid contact surface mapping (OWC / WGC)
- Mineral inversion from observed $V_p/V_s$ ratios

### Carbonate UCS Calibration & Machine Learning
- Systematic benchmarking of 11 published dynamic-to-static correlations
- Zone-specific power-law calibration with bootstrap confidence intervals
- Ensemble ML prediction (Random Forest, Gradient Boosting, XGBoost)
- Honest evaluation protocol: stratified 80/20 split, 5-fold CV (refit inside folds), Leave-One-Well-Out (LOWO)
- Feature importance aligned by name (no positional misalignment)

### Spatial Autocorrelation & Uncertainty
- Lag-1 autocorrelation estimation and effective sample size ($N_{eff}$) calculation
- Moving Block Bootstrap (MBB) at multiple physical block lengths
- Corrected 95% Confidence Intervals (CI) and Prediction Intervals (PI) accounting for spatial dependency

### Publication-Quality Output
- Automatic generation of 1D profiles, 2D contour maps, 3D surfaces, Mohr diagrams, and risk zonation maps
- Dual-unit support (SI: MPa/m + Imperial: psi/ft)
- 15+ manuscript-ready tables and figures per module
- Comprehensive JSON summaries for reproducible reporting

### Data Privacy & Synthetic Generators
- Built-in **Data Privacy Guard (DPG)**: if real field data are absent, physically consistent synthetic datasets are generated automatically
- Autocorrelated AR(1) well-log simulation matching realistic spatial dependency structures
- 100% NDA-compliant: no real well names, coordinates, or field identifiers in any output

---

## 📁 Repository Structure

Geomechanics-Python-Framework/
│
├── modules/
│ ├── geomechanics_1d_mem/ # 1D Mechanical Earth Model
│ │ ├── src/
│ │ │ ├── init.py
│ │ │ ├── config.py
│ │ │ ├── data_loader.py
│ │ │ ├── geomechanics.py
│ │ │ ├── main.py
│ │ │ └── visualization.py
│ │ └── examples/
│ │
│ ├── subsidence_analysis/ # Land Subsidence Module
│ │ ├── src/
│ │ │ ├── geertsma.py
│ │ │ ├── monte_carlo.py
│ │ │ ├── gibson.py
│ │ │ ├── influence_function.py
│ │ │ ├── comparator.py
│ │ │ └── visualization.py
│ │ ├── data/
│ │ └── examples/
│ │
│ ├── reservoir_3d_geomechanics/ # 3D Seismic Geomechanics
│ │ ├── src/
│ │ │ ├── init.py
│ │ │ ├── calibration_3d.py
│ │ │ ├── elastic_properties_3d.py
│ │ │ └── stress_model_3d.py
│ │ ├── data/
│ │ └── examples/
│ │
│ ├── fault_reactivation/ # Fault Slip & Injection Limits
│ │ ├── init.py
│ │ ├── fault_reactivation_analysis.py
│ │ ├── monte_carlo_analysis.py
│ │ └── data/
│ │
│ ├── rock_physics/ # AVO & Fluid Substitution
│ │ ├── init.py
│ │ ├── rock_physics_analysis.py
│ │ ├── fluid_substitution_contacts.py
│ │ └── data/
│ │
│ ├── ucs_calibration/ # Carbonate UCS & ML
│ │ ├── init.py
│ │ ├── ucs_methods_evaluation.py
│ │ ├── ml_ucs_prediction.py
│ │ └── data/
│ │
│ ├── dynamic_to_static_calibration/ # MBB Autocorrelation
│ │ ├── init.py
│ │ ├── stage_a_regression_analysis.py
│ │ ├── stage_b_block_bootstrap.py
│ │ └── data/
│ │
│ └── wellbore_stability_3d/ # Planned v3.5
│ └── init.py
│
├── docs/ # Theory guides & tutorials
├── data/ # Global synthetic data root
│ └── sample/
│
├── generate_synthetic_data.py # Master data orchestrator
├── pyproject.toml # PyPI packaging config
├── requirements.txt # Python dependencies
├── CITATION.cff # Citation metadata
├── .gitignore # Security & privacy filters
├── LICENSE # AGPL-3.0
├── LICENSE-ACADEMIC.md
├── LICENSE-COMMERCIAL.md
└── README.md


---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework.git
cd Geomechanics-Python-Framework

# Install dependencies
pip install -r requirements.txt
2. Generate Synthetic Data (First Run)
The framework includes a Data Privacy Guard (DPG). Running the orchestrator generates physically consistent synthetic datasets across all modules, enabling full pipeline execution without real field data:

python generate_synthetic_data.py

# ── Fault Reactivation & Probabilistic Risk ──
python modules/fault_reactivation/fault_reactivation_analysis.py
python modules/fault_reactivation/monte_carlo_analysis.py

# ── Rock Physics & Gassmann Fluid Substitution ──
python modules/rock_physics/rock_physics_analysis.py
python modules/rock_physics/fluid_substitution_contacts.py

# ── Carbonate UCS Benchmarking & Machine Learning ──
python modules/ucs_calibration/ucs_methods_evaluation.py
python modules/ucs_calibration/ml_ucs_prediction.py

# ── Dynamic-to-Static Calibration & MBB ──
python modules/dynamic_to_static_calibration/stage_a_regression_analysis.py
python modules/dynamic_to_static_calibration/stage_b_block_bootstrap.py

# ── 3D Seismic Geomechanics ──
python modules/reservoir_3d_geomechanics/examples/01_full_3d_geomechanics_workflow.py

# ── Subsidence Analysis ──
python modules/subsidence_analysis/examples/visualization_example.py

💻 Code Snippets
Example 1: Probabilistic Fault Reactivation Assessment

from modules.fault_reactivation.monte_carlo_analysis import (
    resolve_traction_vec, PARAMS_MEAN, SIGMAS
)
import numpy as np

# Sample 10,000 realizations of in-situ stress
N = 10000
Sv = np.random.normal(PARAMS_MEAN['Sv'], SIGMAS['Sv'], N)
SH = np.random.normal(PARAMS_MEAN['SH_max'], SIGMAS['SH_max'], N)
Sh = np.random.normal(PARAMS_MEAN['Sh_min'], SIGMAS['Sh_min'], N)
Pp = np.random.normal(PARAMS_MEAN['Pp'], SIGMAS['Pp'], N)

# Resolve tractions on a fault plane (strike=55°, dip=70°)
sn, tau = resolve_traction_vec(
    strike=55.0, dip=70.0,
    Sv=Sv, SH=SH, Sh=Sh,
    SH_az=35.0, Pp=Pp, alpha=1.0
)

# Compute normalized slip tendency
mu = np.tan(np.deg2rad(np.random.normal(37.6, 2.0, N)))
Ts_norm = (tau / np.maximum(sn, 0.1)) / mu

print(f"Mean Ts_norm: {np.mean(Ts_norm):.3f}")
print(f"P95 Ts_norm:  {np.percentile(Ts_norm, 95):.3f}")
print(f"PoF (Ts>=1):  {np.mean(Ts_norm >= 1.0)*100:.2f}%")

Example 2: Gassmann Fluid Substitution
from modules.rock_physics.fluid_substitution_contacts import gassmann_subst
import numpy as np

# Baseline brine-saturated velocities
Vp_brine = np.array([4500.0, 4200.0, 3800.0])
Vs_brine = np.array([2400.0, 2200.0, 1900.0])
rho_brine = np.array([2.55, 2.48, 2.40])
phi = np.array([0.08, 0.12, 0.18])

# Substitute to 100% gas saturation
Vp_gas, Vs_gas, rho_gas, K_sat = gassmann_subst(
    Vp_in=Vp_brine, Vs_in=Vs_brine, rho_in=rho_brine,
    K_mineral=94.9, G_mineral=45.0,
    K_fl_orig=2.60, K_fl_new=0.035,
    rho_fl_orig=1.10, rho_fl_new=0.25,
    phi=phi
)

print(f"Vp shift (brine→gas): {Vp_gas - Vp_brine} m/s")

Example 3: UCS Machine Learning Prediction
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# Load master dataset (auto-generated or real)
df = pd.read_csv("modules/ucs_calibration/data/master_dataset.csv")
features = ['GR', 'RHOB', 'NPHI', 'DT', 'DTS', 'YME_DYN']
X = df[features]
y = df['UCS_FINAL']

# Train with honest evaluation
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('model', GradientBoostingRegressor(n_estimators=300, max_depth=6))
])
pipe.fit(X.iloc[:800], y.iloc[:800])
score = pipe.score(X.iloc[800:], y.iloc[800:])
print(f"Test R²: {score:.4f}")

⚖️ Licensing
This repository uses a Dual-Licensing Model:

🎓 Academic & Research Use (AGPL-3.0): Free of charge for universities, non-commercial research, and educational purposes. See LICENSE-ACADEMIC.md.
🏭 Commercial License: Required for commercial oil & gas operations, consulting services, and software integrations. Contact: sqaredaqi70@gmail.com. See LICENSE-COMMERCIAL.md.
📄 Citation
If you use this framework or parts of its code in your research or publications, please cite it as:


@software{gharedaghi2025geomech,
  author  = {Gharedaghi, Saeed},
  title   = {Geomechanics Python Framework: A Comprehensive Toolkit for
             Petroleum Geomechanics, 3D Earth Modeling, Fault Reactivation,
             Rock Physics, and Carbonate Characterization},
  year    = {2025},
  version = {3.0.0},
  url     = {https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework},
  license = {AGPL-3.0}
}

👤 Author
Saeed Gharedaghi
Petroleum Geomechanics & Subsurface Engineering

📧 Email: sqaredaqi70@gmail.com
💻 GitHub: @sqaredaqi70-dot

<p align="center"> <sub>Made with dedication for the global geomechanics and petroleum engineering community</sub><br> <sub>Copyright © 2024–2025 Saeed Gharedaghi. All rights reserved.</sub> </p> ```
