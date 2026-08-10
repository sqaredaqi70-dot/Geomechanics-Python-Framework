# 🛢️ Geomechanics Python Framework

<p align="center">
  <b>A Comprehensive Python Framework for Petroleum Geomechanics & Subsurface Engineering</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" /></a>
  <a href="LICENSE-COMMERCIAL.md"><img src="https://img.shields.io/badge/Commercial-Available-green.svg" /></a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" />
  <img src="https://img.shields.io/badge/Platform-Windows_|_Linux_|_macOS-lightgrey.svg" />
</p>

---

## 📖 Overview

**Geomechanics Python Framework** is a comprehensive open-source toolkit designed for petroleum geomechanics analysis and subsurface engineering. It provides end-to-end workflows for:

- **1D Mechanical Earth Modeling (MEM)** for carbonate reservoirs
- **Wellbore stability** and **mud weight window** analysis
- **In-situ stress** characterization (Sv, SHmax, Shmin)
- **Pore pressure** prediction
- **Land subsidence** analysis (upcoming)
- **3D reservoir geomechanics** (upcoming)

Built primarily for **Asmari carbonate reservoirs** (SW Iran) but adaptable to other lithologies and geological settings worldwide.

---

## 📦 Included Modules

| # | Module | Description | Status |
|---|--------|-------------|:------:|
| 1 | 🏔️ **1D MEM** | 1D Mechanical Earth Model for carbonate reservoirs | ✅ v1.0 |
| 2 | 🌍 **Subsidence Analysis** | Land subsidence from reservoir depletion | 🚧 In Development |
| 3 | 🎯 **3D Reservoir Geomechanics** | 3D stress and strain modeling | 📋 Planned |
| 4 | 🔧 **Wellbore Stability 3D** | Advanced 3D wellbore stability | 📋 Planned |

---

## ✨ Key Features

- ✅ **Multi-method UCS estimation** (Horsrud, McNally, CDE, and more)
- ✅ **Dynamic & static elastic moduli** calculation
- ✅ **Stress regime classification** (Normal, Strike-slip, Reverse)
- ✅ **Mohr-Coulomb failure analysis**
- ✅ **Publication-quality visualizations** (2D, 3D, cross-sections)
- ✅ **LAS file support** via `lasio`
- ✅ **Modular architecture** for easy extension
- ✅ **Synthetic sample data** included

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework.git
cd Geomechanics-Python-Framework

# Install dependencies
pip install -r requirements.txt

# Generate synthetic sample data (first-time only)
python generate_synthetic_data.py

# Run the 1D MEM example
python modules/geomechanics_1d_mem/src/main.py
