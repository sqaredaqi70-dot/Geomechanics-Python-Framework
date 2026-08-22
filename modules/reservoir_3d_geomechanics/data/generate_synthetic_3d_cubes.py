"""
Synthetic 3D Seismic Cube Generator
===================================
Generates small synthetic 3D Numpy arrays representing Vp, Vs, Density, and Depth
for an anticline structure to test the 3D Geomechanics Module without large SEG-Y files.
"""

import numpy as np
from pathlib import Path


def generate_synthetic_3d_grid(output_dir: str = "./data/synthetic_3d"):
    """Generates synthetic 3D arrays (Inline x Crossline x Depth)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Dimensions: 50 Inlines x 50 Crosslines x 40 Depth Samples
    n_il, n_xl, n_z = 50, 50, 40

    x = np.linspace(-5000, 5000, n_xl)
    y = np.linspace(-5000, 5000, n_il)
    z = np.linspace(2000, 2500, n_z)

    X, Y = np.meshgrid(x, y)
    Z_3d = np.zeros((n_il, n_xl, n_z))

    for k in range(n_z):
        Z_3d[:, :, k] = z[k] - 150.0 * np.exp(-((X / 3000)**2 + (Y / 3000)**2))

    # Generate synthetic Vp (m/s), Vs (m/s), Density (g/cc)
    vp_3d = 4500.0 + 0.5 * (Z_3d - 2000) + np.random.normal(0, 50, Z_3d.shape)
    vs_3d = vp_3d / 1.732 + np.random.normal(0, 30, Z_3d.shape)
    density_3d = 2.3 + 0.0002 * (Z_3d - 2000) + np.random.normal(0, 0.02, Z_3d.shape)

    np.save(out_path / "vp_3d.npy", vp_3d.astype(np.float32))
    np.save(out_path / "vs_3d.npy", vs_3d.astype(np.float32))
    np.save(out_path / "density_3d.npy", density_3d.astype(np.float32))
    np.save(out_path / "depth_3d.npy", Z_3d.astype(np.float32))

    print(f"✅ Synthetic 3D cubes generated in: {out_path.resolve()}")


if __name__ == "__main__":
    generate_synthetic_3d_grid()
