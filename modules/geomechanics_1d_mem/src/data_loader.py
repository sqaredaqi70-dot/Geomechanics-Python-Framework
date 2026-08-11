                       DATA LOADER MODULE

Utilities for loading LAS files, formation tops, and preprocessing
well log data for geomechanical analysis.

Author:  Saeed Gharedaghi
Contact: sqaredaqi70@gmail.com
License: AGPL-3.0 (Academic) / Commercial (see LICENSE files)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lasio
except ImportError:
    raise ImportError("Please install lasio: pip install lasio")

from . import config as cfg

warnings.filterwarnings('ignore')


# FORMATION TOPS LOADER
def load_formation_tops(tops_file):
    """
    Load formation tops from a formatted text file.

    Parameters
    ----------
    tops_file : str or Path
        Path to the formation tops text file.

    Returns
    -------
    pd.DataFrame
        Columns: Well, Surface, MD, Z
    """
    tops_file = Path(tops_file)
    if not tops_file.exists():
        raise FileNotFoundError(f"Tops file not found: {tops_file}")

    with open(tops_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if 'Well' in line and 'Surface' in line and 'MD' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header line in {tops_file}")

    # Parse data rows
    tops_rows = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                tops_rows.append({
                    'Well':    parts[1],
                    'Surface': parts[2].strip('"\''),
                    'MD':      float(parts[3]),
                    'Z':       float(parts[4]),
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(tops_rows)


# LAS FILE LOADER
def load_las_file(las_path):
    """
    Load a LAS file into a pandas DataFrame.

    Parameters
    ----------
    las_path : str or Path
        Path to the LAS file.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'DEPTH' as the first column.
    """
    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")

    df = lasio.read(str(las_path)).df().reset_index()
    df.rename(columns={df.columns[0]: 'DEPTH'}, inplace=True)
    return df


# UNIT CONVERSION
def convert_kpa_to_mpa(df, columns=None):
    """
    Convert kPa columns to MPa (in-place).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pressure/stress columns.
    columns : list, optional
        List of columns to convert. Uses cfg.STRESS_KPA_COLUMNS if None.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame.
    """
    if columns is None:
        columns = cfg.STRESS_KPA_COLUMNS

    for col in columns:
        if col in df.columns:
            df[col] = df[col] * cfg.KPA_TO_MPA
    return df


# ZONE ASSIGNMENT
def assign_zones(df, well_tops):
    """
    Assign geomechanical zones to depth samples based on well tops.

    Parameters
    ----------
    df : pd.DataFrame
        Well log DataFrame with 'DEPTH' column.
    well_tops : pd.DataFrame
        Formation tops for this well.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'ZONE' column.
    """
    df = df.copy()
    df['ZONE'] = 'Undiff.'

    if len(well_tops) == 0:
        return df

    # Build list of (zone_top_depth, zone_name)
    zone_boundaries = []
    for zone_name, markers in cfg.ZONE_MARKERS.items():
        for marker in markers:
            match = well_tops[well_tops['Surface'] == marker]
            if len(match) > 0:
                zone_boundaries.append(
                    (float(match['MD'].iloc[0]), zone_name)
                )
                break

    zone_boundaries.sort(key=lambda x: x[0])

    # Assign zones
    for i, (mt, zn) in enumerate(zone_boundaries):
        if i < len(zone_boundaries) - 1:
            mask = ((df['DEPTH'] >= mt) &
                    (df['DEPTH'] < zone_boundaries[i + 1][0]))
        else:
            mask = df['DEPTH'] >= mt
        df.loc[mask, 'ZONE'] = zn

    return df


# DENSITY COLUMN FINDER
def find_density_column(df):
    """
    Find the best density column in a DataFrame.

    Priority order: composite density variants first, then individual.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing density columns.

    Returns
    -------
    str or None
        Column name of best density curve, or None.
    """
    if df is None:
        return None

    priority = [
        'RHOB',
        'DEN_COM_AMO', 'DEN_COM_EXT', 'DEN_COM_WEN', 'DEN_COM_MIL',
        'DEN_AMOCO', 'DEN_EXTRAPOLATE', 'DEN_MILLER',
        'DEN_WENDT_NON_ACOUSTIC',
    ]

    for col in priority:
        if col not in df.columns:
            continue
        vals = df[col].replace(0, np.nan).dropna()
        if len(vals) > 100 and 1.5 < float(vals.median()) < 3.2:
            return col

    return None


# WELL LOADER (MAIN FUNCTION)
def load_well(well_name, wire_las_path, ovb_las_path=None,
              tops_df=None):
    """
    Load and preprocess data for a single well.

    Parameters
    ----------
    well_name : str
        Name of the well (e.g., 'Well_A').
    wire_las_path : str or Path
        Path to the wireline LAS file.
    ovb_las_path : str or Path, optional
        Path to the overburden LAS file (density, Sv).
    tops_df : pd.DataFrame, optional
        Formation tops DataFrame.

    Returns
    -------
    dict
        Well data dictionary with all preprocessed logs.
    """
    # Load wireline LAS
    df_wire = load_las_file(wire_las_path)
    df_wire = convert_kpa_to_mpa(df_wire)

    # Load overburden LAS if available
    df_ovb = None
    rho_col = None
    if ovb_las_path is not None and Path(ovb_las_path).exists():
        df_ovb = load_las_file(ovb_las_path)
        rho_col = find_density_column(df_ovb)

    # Get well tops
    if tops_df is not None:
        well_tops = tops_df[tops_df['Well'] == well_name]
    else:
        well_tops = pd.DataFrame()

    # Find reservoir top/base
    top_row = well_tops[well_tops['Surface'] == cfg.RESERVOIR_TOP]
    bot_row = well_tops[well_tops['Surface'] == cfg.RESERVOIR_BASE]

    if len(top_row) == 0 or len(bot_row) == 0:
        # Fallback: use full LAS range
        top_md = float(df_wire['DEPTH'].min())
        bot_md = float(df_wire['DEPTH'].max())
    else:
        top_md = float(top_row['MD'].iloc[0])
        bot_md = float(bot_row['MD'].iloc[0])

    # Filter to reservoir interval
    df = df_wire[
        (df_wire['DEPTH'] >= top_md) & (df_wire['DEPTH'] <= bot_md)
    ].copy()

    # Assign zones
    df = assign_zones(df, well_tops)

    # Add depth in feet
    df['DEPTH_ft'] = df['DEPTH'] * cfg.M_TO_FT

    # Convert stress columns to psi
    for src, dst in [
        ('SHMAX_PHS', 'SH_psi'),
        ('SHMIN_PHS', 'Sh_psi'),
        ('FINAL_PP',  'Pp_psi'),
    ]:
        if src in df.columns:
            df[dst] = df[src] * cfg.MPA_TO_PSI

    # Density lookup from OVB file
    if df_ovb is not None and rho_col is not None:
        rho_data = df_ovb[['DEPTH', rho_col]].dropna()
        rho_data = rho_data.sort_values('DEPTH').reset_index(drop=True)
        df = pd.merge_asof(
            df.sort_values('DEPTH').reset_index(drop=True),
            rho_data,
            on='DEPTH',
            tolerance=5.0,
            direction='nearest',
        )

    return {
        'name':     well_name,
        'df':       df,
        'df_ovb':   df_ovb,
        'rho_col':  rho_col,
        'top_md':   top_md,
        'bot_md':   bot_md,
        'tops':     well_tops,
        'position': cfg.WELL_POSITIONS.get(well_name, (0.0, 0.0)),
    }


# MULTI-WELL LOADER
def load_all_wells(data_dir=None, tops_file=None):
    """
    Load all wells configured in cfg.WELL_NAMES.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing LAS files. Defaults to cfg.DATA_DIR.
    tops_file : str or Path, optional
        Path to formation tops file.

    Returns
    -------
    dict
        Dictionary of {well_name: well_data_dict}.
    """
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    data_dir = Path(data_dir)

    # Load tops
    tops_df = None
    if tops_file is None:
        tops_candidates = list(data_dir.glob("*Tops*.txt"))
        if tops_candidates:
            tops_file = tops_candidates[0]

    if tops_file is not None and Path(tops_file).exists():
        tops_df = load_formation_tops(tops_file)

    # Load wells
    wells = {}
    for well_name in cfg.WELL_NAMES:
        # Look for synthetic files first, then real
        wire_candidates = [
            data_dir / f"{well_name}_synthetic.las",
            data_dir / f"{well_name}_WIRE.las",
            data_dir / f"{well_name}.las",
        ]
        wire_path = None
        for c in wire_candidates:
            if c.exists():
                wire_path = c
                break

        if wire_path is None:
            print(f"  ⚠ {well_name}: No LAS file found, skipping.")
            continue

        # Optional overburden file
        ovb_candidates = [
            data_dir / f"{well_name}_WIRE_OVB.las",
            data_dir / f"{well_name}_OVB.las",
        ]
        ovb_path = None
        for c in ovb_candidates:
            if c.exists():
                ovb_path = c
                break

        wells[well_name] = load_well(
            well_name, wire_path, ovb_path, tops_df
        )
        n = len(wells[well_name]['df'])
        print(f"  ✅ {well_name}: {n:,} samples loaded")

    return wells
