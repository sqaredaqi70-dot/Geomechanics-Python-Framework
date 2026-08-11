# 🏔️ 1D Mechanical Earth Model (MEM)

Module for 1D Mechanical Earth Modeling of carbonate reservoirs.

## Overview

This module implements a complete 1D MEM workflow, including:

- **Multi-method UCS estimation**
  - Horsrud (2001)
  - McNally (1987)
  - CDE method
  - Sonic-based correlations
  - Elastic modulus-based estimates

- **Elastic moduli calculation**
  - Dynamic Young's modulus
  - Static Young's modulus (multiple correlations)
  - Poisson's ratio (dynamic and static)
  - Shear and bulk moduli

- **In-situ stress analysis**
  - Vertical stress (Sv) from density integration
  - Maximum horizontal stress (SHmax)
  - Minimum horizontal stress (Shmin)
  - Stress regime classification

- **Pore pressure prediction**

- **Wellbore stability**
  - Collapse pressure
  - Fracture pressure
  - Safe mud weight window
  - Moderated collapse (with safety margin)

- **Visualization**
  - 1D depth profiles per well
  - Cross-plots and correlations
  - Stress regime polygons
  - Mohr-Coulomb envelopes
  - 3D property visualization

## Usage

### With Sample Data

```bash
# From project root
python modules/geomechanics_1d_mem/src/main.py
With Your Own Data
Update paths in src/config.py
Place LAS files in data/raw/
Update well names in src/config.py
Run: python src/main.py
Directory Structure
text

geomechanics_1d_mem/
├── README.md              This file
├── src/                   Source code
│   ├── config.py          Configuration
│   ├── data_loader.py     LAS file reader
│   ├── geomechanics.py    Core calculations
│   ├── visualization.py   Plotting utilities
│   └── main.py            Main workflow
├── data/
│   └── sample/            Sample data (auto-generated)
├── notebooks/             Jupyter examples
└── docs/                  Module documentation
Methodology
The workflow follows industry-standard practices for building 1D MEMs
in carbonate reservoirs. Key references:

UCS estimation: Horsrud (2001), McNally (1987)
Elastic moduli: Fjaer et al. (2008)
Stress analysis: Zoback (2007)
Wellbore stability: Aadnoy & Looyeh (2019)
Output
The module generates:

Per-well 1D MEM profiles (9-track composite logs)
Cross-plots for property relationships
UCS method evaluation and ranking
Stress regime characterization
Mohr-Coulomb failure analysis
Wellbore stability charts
Multi-well comparison plots
3D visualization of properties
Output is saved to output/figures/.

Requirements
See main requirements.txt in project root.

Key dependencies:

numpy
pandas
matplotlib
scipy
lasio
