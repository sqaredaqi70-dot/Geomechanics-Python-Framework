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

System Requirements
Python: 3.9 or higher
RAM: 4 GB minimum (8 GB recommended for 3D visualization)
OS: Windows, Linux, or macOS
📊 Data Availability Statement
The real field data used for developing this framework cannot be shared due to confidentiality agreements with data providers. However, this repository provides:

✅ Synthetic sample data that mimics realistic wireline log characteristics
✅ Complete workflow code applicable to any similar dataset
✅ Detailed methodology documentation for full reproducibility of the approach
The synthetic data allows users to test and understand the workflow, though absolute values will naturally differ from real-field applications.

⚖️ Licensing
This project uses a dual-licensing model to serve both academic and commercial users:

🎓 Free for Academic & Research Use (AGPL-3.0)
Free of charge for:

Universities and research institutes
Individual researchers and students
Non-commercial academic research
Educational purposes
📄 See LICENSE-ACADEMIC.md for detailed terms.

🏭 Commercial License Required for Industrial Use
Any commercial or industrial use requires a separate paid license, including:

Oil & gas operations (national/international oil companies)
Consulting services and contractors
Integration into commercial software
Use by for-profit organizations
📄 See LICENSE-COMMERCIAL.md for details.

📧 Commercial licensing inquiries: sqaredaqi70@gmail.com
📄 Citation
If you use this framework in your research, please cite it as follows:

bibtex

@software{aqi2024geomechframework,
  author  = {Aqi, Sqared},
  title   = {Geomechanics Python Framework: A Comprehensive Toolkit
             for Petroleum Geomechanics and Subsurface Engineering},
  year    = {2024},
  version = {1.0.0},
  url     = {https://github.com/sqaredaqi70-dot/Geomechanics-Python-Framework},
  license = {AGPL-3.0}
}
👤 Author
Saeed Gharedaghi

📧 Email: sqaredaqi70@gmail.com
💻 GitHub: @sqaredaqi70-dot
🎓 Field: Petroleum Geomechanics & Reservoir Engineering
🤝 Contributing
Contributions are welcome and greatly appreciated! Please:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
🐛 Bug Reports & Feature Requests
Found a bug or have a feature idea? Please open an issue.

⚠️ Disclaimer
This software is provided "as is" for research and educational purposes without warranty of any kind. For operational drilling, completion, or reservoir management decisions, please consult qualified petroleum engineers and geoscientists. The author assumes no liability for damages arising from the use of this software.

🌟 Support the Project
If you find this framework useful for your work, please consider:

⭐ Starring the repository
🍴 Forking and contributing
📢 Sharing with your colleagues
📅 Roadmap
 v1.0 — 1D MEM for carbonate reservoirs
 v1.1 — Enhanced UCS methods and additional lithologies
 v2.0 — Land subsidence analysis module
 v2.5 — 3D reservoir geomechanics
 v3.0 — Advanced 3D wellbore stability with deviation
<p align="center"> <sub>Made with ❤️ for the geomechanics community</sub><br> <sub>Copyright © 2024 Saeed Gharedaghi. All rights reserved.</sub> </p> ```
