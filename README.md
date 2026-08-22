# 🌋 Geomechanics Python Framework

A Comprehensive Python Framework for Petroleum Geomechanics, 1D Mechanical Earth Modeling (MEM), 3D Reservoir Stress Characterization & Production-Induced Subsidence Analysis.

![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Commercial License](https://img.shields.io/badge/License-Commercial_Available-gold.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![Framework Version](https://img.shields.io/badge/Version-v2.5.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Active_Development-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows_%7C_Linux_%7C_macOS-lightgrey.svg)

---

## 📖 Overview

**Geomechanics Python Framework** is an open-source, modular, and extensible software toolkit designed for subsurface engineering, petroleum geomechanics, and reservoir management. It integrates well-log processing, 1D MEM construction, 3D seismic-driven geomechanical modeling, and time-dependent/probabilistic land subsidence predictions into unified workflows.

Although tailored and calibrated for **carbonate reservoirs** (e.g., Asmari Formation in SW Iran), the framework is fully parameterizable and adaptable to any geological setting worldwide (sandstones, shales, chalks).

---

## 📦 Included Modules & Roadmap

| # | Module | Description | Status | Version |
|---|--------|-------------|--------|---------|
| 1 | 🏔️ **1D MEM** | 1D Mechanical Earth Modeling (Elastic moduli, UCS, In-situ stresses, Mohr-Coulomb) | ✅ Complete | `v1.0` |
| 2 | 🌍 **Subsidence Analysis** | Analytical (Geertsma), Probabilistic (Monte Carlo, Sobol/Tornado), Consolidation (Gibson), and Spatial (DIF/Nucleus of Strain) models | ✅ Complete | `v2.0` |
| 3 | 🎯 **3D Reservoir Geomechanics** | Seismic-driven 3D properties ($E, \nu, \text{UCS}$) with dynamic-to-static well calibration & 3D stress field ($S_v, P_p, S_{hmin}, S_{Hmax}$) | ✅ Complete | `v2.5` |
| 4 | 🔧 **Wellbore Stability 3D** | Trajectory optimization, breakout/tensile fracture risk, and mud weight window | 📋 Planned | `v3.0` |

---

## ✨ Key Features

- **1D Earth Modeling:** Multi-method UCS estimation (Horsrud, McNally, CDE, etc.), elastic moduli calculation, and stress regime classification.
- **Advanced Subsidence Engine:**
  - **Geertsma (1973):** Analytical poroelastic solution for disk-shaped reservoirs.
  - **Monte Carlo Uncertainty:** Probabilistic simulations with truncated distributions, Tornado sensitivity, and Sobol variance-based indices ($S_1, S_T$).
  - **Gibson / Terzaghi Consolidation:** Time-dependent subsidence coupled with $1\text{D}$ consolidation theory and $C_v$ sensitivity.
  - **Depth Influence Function (DIF):** 2D spatial convolution via Nucleus of Strain kernel.
  - **Multi-Model Comparator:** Benchmarking tool for cross-method validation.
- **3D Seismic Geomechanics:** Direct integration of 3D seismic inversion cubes ($V_p, V_s, \rho$) with well-based dynamic-to-static calibrations ($E_{dyn} \to E_{static} \to \text{UCS}$) and scale-factor corrections.
- **Publication-Quality Visualizations:** Automatic generation of 1D profiles, 2D contour maps, 3D deformed surfaces, risk zonation maps, and interactive plots.
- **Data Privacy & Synthetic Generators:** Includes scripts to generate realistic 1D, 2D, and 3D synthetic datasets to test workflows without violating NDAs or confidentiality agreements.

---

## 📁 Repository Structure
Geomechanics-Python-Framework/
│
├── modules/
│ ├── geomechanics_1d_mem/ # 1D Mechanical Earth Model
│ │ ├── src/
│ │ └── examples/
│ │
│ ├── subsidence_analysis/ # Comprehensive Subsidence Module
│ │ ├── src/
│ │ │ ├── geertsma.py # Analytical model
│ │ │ ├── monte_carlo.py # Uncertainty & Sensitivity
│ │ │ ├── gibson.py # Consolidation model
│ │ │ ├── influence_function.py # DIF (Nucleus of Strain)
│ │ │ ├── comparator.py # Multi-method benchmarking
│ │ │ └── visualization.py # Publication-ready plotting
│ │ ├── data/ # Synthetic 2D grid generator
│ │ └── examples/ # Executable example workflows
│ │
│ └── reservoir_3d_geomechanics/ # 3D Seismic-Driven Geomechanics
│ ├── src/
│ │ ├── calibration_3d.py # Dynamic-to-static calibration
│ │ ├── elastic_properties_3d.py# 3D Moduli calculations
│ │ └── stress_model_3d.py # 3D Stress fields
│ ├── data/ # Synthetic 3D cube generator
│ └── examples/ # 3D workflow runner
│
├── docs/ # Getting started and theory guides
└── generate_synthetic_data.py # Top-level synthetic data generator

text


---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework.git
cd Geomechanics-Python-Framework

# Install dependencies
pip install -r requirements.txt
2. Running Examples
Subsidence & Sensitivity Workflow:
Bash

python modules/subsidence_analysis/examples/visualization_example.py
3D Seismic Geomechanics Workflow:
Bash

python modules/reservoir_3d_geomechanics/examples/01_full_3d_geomechanics_workflow.py
💻 Code Snippets
Example: Running a Probabilistic Subsidence Assessment
Python

from modules.subsidence_analysis.src import MCConfig, MonteCarloSubsidence, tornado_analysis

# Configure Monte Carlo simulation
config = MCConfig(n_iterations=10000, random_seed=42)
engine = MonteCarloSubsidence(config)

# Run for depletion scenarios
results = engine.run_all_scenarios([5, 10, 15, 20])
df_summary = engine.get_summary_dataframe()
print(df_summary)

# Tornado Sensitivity
tornado_df = tornado_analysis(config, depletion_mpa=10.0)
print(tornado_df)
⚖️ Licensing
This repository uses a Dual-Licensing model:

🎓 Academic & Research Use (AGPL-3.0): Free of charge for universities, non-commercial research, and educational purposes. See LICENSE-ACADEMIC.md.
🏭 Commercial License: Required for commercial oil & gas operations, consulting services, and software integrations. Contact: sqaredaqi70@gmail.com. See LICENSE-COMMERCIAL.md.
📄 Citation
If you use this framework or parts of its code in your research or publications, please cite it as:

bibtex

@software{gharedaghi2024geomech,
  author  = {Gharedaghi, Saeed},
  title   = {Geomechanics Python Framework: A Comprehensive Toolkit for Petroleum Geomechanics, 3D Earth Modeling, and Reservoir Subsidence},
  year    = {2024},
  version = {2.5.0},
  url     = {https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework},
  license = {AGPL-3.0}
}
👤 Author
Saeed Gharedaghi
Petroleum Geomechanics & Subsurface Engineering
📧 Email: sqaredaqi70@gmail.com
💻 GitHub: @sqaredaqi70-dot

<p align="center"> <sub>Made with ❤️ for the global geomechanics and petroleum engineering community</sub><br> <sub>Copyright © 2024-2025 Saeed Gharedaghi. All rights reserved.</sub> </p> ```
