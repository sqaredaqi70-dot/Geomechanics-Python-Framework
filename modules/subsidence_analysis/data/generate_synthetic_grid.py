"""
Synthetic Reservoir Grid Generator
Generates non-confidential, realistic structural grids for subsidence analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_surfaces(output_dir: str):
    """Generates synthetic top and bottom horizons for a generic carbonate anticline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Grid dimensions (e.g., 24km x 24km)
    x = np.linspace(-12000, 12000, 150)
    y = np.linspace(-12000, 12000, 150)
    X, Y = np.meshgrid(x, y)
    
    # Generic Anticline formula
    R = np.sqrt(X**2 + (Y/1.5)**2)
    top_depth = 2275 - 300 * np.exp(-(R/5000)**2) + np.random.normal(0, 5, X.shape)
    
    # Constant-ish thickness with slight thinning at edges
    thickness = 107 + 20 * np.exp(-(R/6000)**2)
    base_depth = top_depth + thickness
    
    # Create mask (e.g., Oil-Water Contact at 2350m)
    mask = top_depth < 2350
    top_depth[~mask] = np.nan
    base_depth[~mask] = np.nan
    
    # Save as simple CSVs (easier for open-source users than CPS-3 format)
    df_top = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Z': top_depth.flatten()}).dropna()
    df_base = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Z': base_depth.flatten()}).dropna()
    
    df_top.to_csv(out_path / "synthetic_top_reservoir.csv", index=False)
    df_base.to_csv(out_path / "synthetic_base_reservoir.csv", index=False)
    
    print(f"✅ Synthetic grids generated in {out_path}")

if __name__ == "__main__":
    generate_synthetic_surfaces("./data/synthetic")
