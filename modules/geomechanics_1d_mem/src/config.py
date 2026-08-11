"""
═══════════════════════════════════════════════════════════════════════
                    CONFIGURATION MODULE
═══════════════════════════════════════════════════════════════════════

Centralized configuration for the 1D Mechanical Earth Model workflow.

All well names, paths, and coordinates are anonymized for public release.
Users applying this workflow to real data should update these values
in their local copy (do not commit real data or paths to public repos).

Author:  Sqared Aqi
Contact: sqaredaqi70@gmail.com
License: AGPL-3.0 (Academic) / Commercial (see LICENSE files)
═══════════════════════════════════════════════════════════════════════
"""

from pathlib import Path


# ══════════════════════════════════════════════════════════════
# PROJECT PATHS
# ══════════════════════════════════════════════════════════════
# Locate project root (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data directories
DATA_DIR   = PROJECT_ROOT / "data" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures_3D").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures_combined").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# WELL CONFIGURATION (ANONYMIZED)
# ══════════════════════════════════════════════════════════════
# Wells are labeled generically to preserve confidentiality.
# For real-data applications, update these lists locally.

WELL_NAMES = ['Well_A', 'Well_B', 'Well_C']

# Relative well positions in kilometers (arbitrary reference frame)
# These are NOT real geographic coordinates
WELL_POSITIONS = {
    'Well_A': (0.0,  0.0),
    'Well_B': (1.7,  0.3),
    'Well_C': (-1.2, -2.8),
}


# ══════════════════════════════════════════════════════════════
# GEOLOGICAL ZONES
# ══════════════════════════════════════════════════════════════
ZONE_ORDER = ['Ghar', 'Asmari-A', 'Asmari-B', 'Jahrum']

ZONE_MARKERS = {
    'Ghar':     ['Ghar_C.R.', 'Ghar_1', 'Ghar_2', 'Ghar_3_1'],
    'Asmari-A': ['Asmari_A', 'AA_Shale'],
    'Asmari-B': ['Asmari_B1', 'Asmari_B2_1', 'Asmari_B3'],
    'Jahrum':   ['Jahrum'],
}

RESERVOIR_TOP  = 'Ghar_C.R.'
RESERVOIR_BASE = 'Pabdeh'


# ══════════════════════════════════════════════════════════════
# ENGINEERING UNIT CONVERSIONS
# ══════════════════════════════════════════════════════════════
G_MS2         = 9.81         # Gravity acceleration (m/s²)
MPA_TO_PSI    = 145.0377     # MPa → psi
GPA_TO_MPSI   = 0.145038     # GPa → Mpsi
M_TO_FT       = 3.28084      # meters → feet
KPA_TO_MPA    = 0.001        # kPa → MPa
GCC_TO_PPG    = 8.345404     # g/cc → ppg
PPG_TO_PSIFT  = 0.052        # ppg → psi/ft


# ══════════════════════════════════════════════════════════════
# MODEL ASSUMPTIONS
# ══════════════════════════════════════════════════════════════
RHO_WATER          = 1.025   # Formation water density (g/cc)
RHO_FALLBACK       = 2.4     # Fallback density if RHOB missing (g/cc)
T0_FRACTION        = 0.10    # Tensile strength = T0_FRACTION × UCS
MW_SAFETY_MARGIN   = 0.5     # Mud weight safety margin (ppg)
SHMAX_AZIMUTH      = 35.0    # Regional SHmax azimuth (degrees)


# ══════════════════════════════════════════════════════════════
# STRESS AND STRENGTH COLUMN NAMES (kPa in LAS files)
# ══════════════════════════════════════════════════════════════
STRESS_KPA_COLUMNS = [
    'UCS_FINAL', 'UCS_FINAL_70', 'UCS_FINAL_80', 'UCS_FINAL_90',
    'UCS_FINAL_110', 'UCS_FINAL_120',
    'UCS_HORSRUD', 'UCS_MCNALLY', 'UCS_CDE',
    'UCS_SMG_RPC', 'UCS_SND_RPC', 'UCS_YME',
    'TSTR_8_4%', 'TSTR_10%',
    'SHMAX_MC_UB', 'SHMAX_PHS', 'SHMIN_MC_LB', 'SHMIN_PHS',
    'FINAL_PP', 'FINAL_PP_COR',
    'COHESION_FROM_GR', 'COHESION_FROM_PLUMB',
]


# ══════════════════════════════════════════════════════════════
# UCS ESTIMATION METHODS
# ══════════════════════════════════════════════════════════════
UCS_METHODS = {
    'UCS_HORSRUD': ('Horsrud (2001)',   '#1f77b4'),
    'UCS_MCNALLY': ('McNally (1987)',   '#ff7f0e'),
    'UCS_CDE':     ('CDE method',       '#2ca02c'),
    'UCS_SMG_RPC': ('Shear modulus',    '#d62728'),
    'UCS_SND_RPC': ('Sonic-based',      '#9467bd'),
    'UCS_YME':     ('E-based',          '#8c564b'),
}


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════
WELL_COLORS = {
    'Well_A': '#e41a1c',  # red
    'Well_B': '#377eb8',  # blue
    'Well_C': '#4daf4a',  # green
}

WELL_MARKERS = {
    'Well_A': 'o',
    'Well_B': 's',
    'Well_C': '^',
}

ZONE_COLORS = {
    'Ghar':     '#2ca02c',
    'Asmari-A': '#d62728',
    'Asmari-B': '#1f77b4',
    'Jahrum':   '#ff7f0e',
    'Undiff.':  'lightgray',
}

ZONE_FACECOLORS = {
    'Ghar':     '#e8f5e9',
    'Asmari-A': '#ffebee',
    'Asmari-B': '#e3f2fd',
    'Jahrum':   '#fff9c4',
    'Undiff.':  '#f5f5f5',
}

FONT_SIZES = {
    'title':   40,
    'label':   36,
    'tick':    26,
    'legend':  30,
    'zone':    26,
    'well':    36,
    'annot':   26,
    'regime':  30,
}
