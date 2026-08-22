"""
Generates the Master Demo Jupyter Notebook (01_Master_Geomechanics_Demo.ipynb)
"""

import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🌋 Geomechanics Python Framework - Master Demo\n",
    "Welcome to the interactive demonstration of the **Geomechanics Python Framework**.\n",
    "This notebook covers:\n",
    "1. **1D MEM & Mud Weight Window**\n",
    "2. **3D Reservoir Geomechanics & Stress Field**\n",
    "3. **Production-Induced Subsidence (Geertsma, Monte Carlo, Gibson)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from modules.subsidence_analysis.src import (\n",
    "    calculate_max_subsidence, MCConfig, MonteCarloSubsidence, plot_radial_profile\n",
    ")\n",
    "from modules.wellbore_stability_3d.src import calculate_mud_weight_window\n",
    "\n",
    "print('✅ Geomechanics Framework successfully imported!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Wellbore Stability Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute Mud Weight Window for a vertical well\n",
    "mww = calculate_mud_weight_window(\n",
    "    depth_m=2500, sv_mpa=55, shmax_mpa=60, shmin_mpa=45, pp_mpa=26, ucs_mpa=65\n",
    ")\n",
    "print(f\"Breakout Mud Weight: {mww['breakout_mw_sg']:.2f} g/cm³\")\n",
    "print(f\"Tensile Mud Weight:  {mww['tensile_mw_sg']:.2f} g/cm³\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Subsidence Radial Profile & Monte Carlo Simulation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "r_km = np.linspace(0, 12, 100)\n",
    "sub_cm = [calculate_max_subsidence(10, 107, 2275, 5000, 27.6, 0.23) / (1 + (rk/5)**2)**0.5 for rk in r_km]\n",
    "\n",
    "fig = plot_radial_profile(r_km, np.array(sub_cm))\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": { "name": "python" }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("./examples/01_Master_Geomechanics_Demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("✅ Created './examples/01_Master_Geomechanics_Demo.ipynb' successfully!")
