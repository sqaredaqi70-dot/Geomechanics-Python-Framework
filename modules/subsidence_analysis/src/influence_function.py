"""
Depth Influence Function (DIF) Module
=====================================
Calculates surface subsidence using the Nucleus of Strain (Influence Function)
method based on Geertsma (1973) and Segall (1992). This method performs a 2D 
spatial integration over the heterogeneous compaction grid.

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
from typing import Union, Tuple


def nucleus_of_strain_kernel(
    r_distance_m: Union[float, np.ndarray], 
    depth_m: float, 
    poisson_ratio: float = 0.25
) -> Union[float, np.ndarray]:
    """
    Computes the point-source influence kernel for a poroelastic half-space.

    Parameters
    ----------
    r_distance_m : float or np.ndarray
        Radial horizontal distance from the source point to the observation point (m).
    depth_m : float
        Depth of the source point (m).
    poisson_ratio : float, optional
        Poisson's ratio of the overburden, by default 0.25.

    Returns
    -------
    float or np.ndarray
        Kernel value (dimensionless geometric modifier).
    """
    r2_d2 = r_distance_m**2 + depth_m**2
    # To avoid division by zero if source is at the exact surface (unlikely but safe)
    r2_d2 = np.maximum(r2_d2, 1e-6)
    
    kernel = ((1.0 - poisson_ratio) / (2.0 * np.pi)) * depth_m / np.power(r2_d2, 1.5)
    return kernel


def compute_dif_subsidence(
    compaction_grid_m: np.ndarray,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    depth_m: float,
    poisson_ratio: float = 0.25,
    step: int = 1
) -> np.ndarray:
    """
    Computes spatial subsidence by convolving the compaction grid with the 
    influence function kernel.

    Parameters
    ----------
    compaction_grid_m : np.ndarray
        2D array of reservoir compaction (Cm * h_eff * dP) in meters.
        Invalid cells should be np.nan.
    x_grid_m : np.ndarray
        2D array of X coordinates (Easting) in meters.
    y_grid_m : np.ndarray
        2D array of Y coordinates (Northing) in meters.
    depth_m : float
        Mean depth of the reservoir in meters.
    poisson_ratio : float, optional
        Poisson's ratio, by default 0.25.
    step : int, optional
        Subsampling step for faster computation. 1 means full resolution, 
        2 computes every 2nd cell and interpolates (much faster for large grids).

    Returns
    -------
    np.ndarray
        2D array of calculated surface subsidence in meters.
    """
    ny, nx = x_grid_m.shape
    
    # Calculate cell area (assuming regular grid)
    dx = np.nanmedian(np.abs(np.diff(x_grid_m, axis=1)))
    dy = np.nanmedian(np.abs(np.diff(y_grid_m, axis=0)))
    cell_area = dx * dy
    
    # Create mask for valid reservoir cells
    valid_mask = np.isfinite(compaction_grid_m)
    xc = x_grid_m[valid_mask]
    yc = y_grid_m[valid_mask]
    compaction_vals = compaction_grid_m[valid_mask]
    
    # Setup calculation grid based on step size
    y_idx = np.arange(0, ny, step)
    x_idx = np.arange(0, nx, step)
    subsidence_coarse = np.full((len(y_idx), len(x_idx)), np.nan)
    
    # Spatial integration (Convolution)
    for iyc, iy in enumerate(y_idx):
        for ixc, ix in enumerate(x_idx):
            if not valid_mask[iy, ix]:
                continue
            
            # Calculate radial distance from current observation point to all source cells
            r = np.sqrt((xc - x_grid_m[iy, ix])**2 + (yc - y_grid_m[iy, ix])**2)
            
            # Avoid singularity at the cell itself (use a fraction of cell size)
            r = np.maximum(r, dx * 0.1)
            
            # Apply kernel and integrate
            kernel_vals = nucleus_of_strain_kernel(r, depth_m, poisson_ratio)
            subsidence_coarse[iyc, ixc] = np.sum(compaction_vals * kernel_vals * cell_area)
            
    # Upscale back to original resolution if step > 1
    if step > 1:
        subsidence_full = np.repeat(np.repeat(subsidence_coarse, step, axis=0), step, axis=1)[:ny, :nx]
    else:
        subsidence_full = subsidence_coarse
        
    subsidence_full[~valid_mask] = np.nan
    return subsidence_full
