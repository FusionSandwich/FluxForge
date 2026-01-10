"""
NIST Standard Material Compositions

Provides standard material compositions from NIST and PNNL Compendium
for use in radiation transport and attenuation calculations.

Epic R - Becquerel Parity

References:
    NIST XCOM: https://www.nist.gov/pml/xcom-photon-cross-sections-database
    PNNL Compendium: https://compendium.cwmd.pnnl.gov
    NIST SRM: https://www.nist.gov/srm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Material:
    """
    Standard material definition with composition.
    
    Attributes
    ----------
    name : str
        Material name
    density : float
        Density in g/cm³
    composition : dict
        Element composition as {symbol: weight_fraction}
    formula : str, optional
        Chemical formula
    source : str
        Data source
    """
    name: str
    density: float
    composition: Dict[str, float]
    formula: str = ""
    source: str = "NIST"
    
    @property
    def elements(self) -> List[str]:
        """List of elements in material."""
        return list(self.composition.keys())
    
    @property
    def weight_fractions(self) -> np.ndarray:
        """Array of weight fractions."""
        return np.array(list(self.composition.values()))
    
    def validate(self) -> bool:
        """Check if weight fractions sum to ~1.0."""
        total = sum(self.composition.values())
        return abs(total - 1.0) < 0.001


# =============================================================================
# NIST Elemental Data (Table 1)
# =============================================================================

NIST_ELEMENTS = {
    'H': {'Z': 1, 'A': 1.00794, 'density': 8.375e-5},
    'He': {'Z': 2, 'A': 4.00260, 'density': 1.663e-4},
    'Li': {'Z': 3, 'A': 6.941, 'density': 0.534},
    'Be': {'Z': 4, 'A': 9.01218, 'density': 1.848},
    'B': {'Z': 5, 'A': 10.811, 'density': 2.370},
    'C': {'Z': 6, 'A': 12.0107, 'density': 2.000},  # Graphite
    'N': {'Z': 7, 'A': 14.0067, 'density': 1.165e-3},
    'O': {'Z': 8, 'A': 15.9994, 'density': 1.332e-3},
    'F': {'Z': 9, 'A': 18.9984, 'density': 1.580e-3},
    'Ne': {'Z': 10, 'A': 20.1797, 'density': 8.385e-4},
    'Na': {'Z': 11, 'A': 22.9898, 'density': 0.971},
    'Mg': {'Z': 12, 'A': 24.3050, 'density': 1.740},
    'Al': {'Z': 13, 'A': 26.9815, 'density': 2.699},
    'Si': {'Z': 14, 'A': 28.0855, 'density': 2.330},
    'P': {'Z': 15, 'A': 30.9738, 'density': 2.200},  # White P
    'S': {'Z': 16, 'A': 32.065, 'density': 2.000},
    'Cl': {'Z': 17, 'A': 35.453, 'density': 2.995e-3},
    'Ar': {'Z': 18, 'A': 39.948, 'density': 1.662e-3},
    'K': {'Z': 19, 'A': 39.0983, 'density': 0.862},
    'Ca': {'Z': 20, 'A': 40.078, 'density': 1.550},
    'Sc': {'Z': 21, 'A': 44.9559, 'density': 2.989},
    'Ti': {'Z': 22, 'A': 47.867, 'density': 4.540},
    'V': {'Z': 23, 'A': 50.9415, 'density': 6.110},
    'Cr': {'Z': 24, 'A': 51.9961, 'density': 7.180},
    'Mn': {'Z': 25, 'A': 54.9380, 'density': 7.440},
    'Fe': {'Z': 26, 'A': 55.845, 'density': 7.874},
    'Co': {'Z': 27, 'A': 58.9332, 'density': 8.900},
    'Ni': {'Z': 28, 'A': 58.6934, 'density': 8.902},
    'Cu': {'Z': 29, 'A': 63.546, 'density': 8.960},
    'Zn': {'Z': 30, 'A': 65.38, 'density': 7.133},
    'Ga': {'Z': 31, 'A': 69.723, 'density': 5.904},
    'Ge': {'Z': 32, 'A': 72.64, 'density': 5.323},
    'As': {'Z': 33, 'A': 74.9216, 'density': 5.730},
    'Se': {'Z': 34, 'A': 78.96, 'density': 4.500},
    'Br': {'Z': 35, 'A': 79.904, 'density': 3.120},  # Liquid
    'Kr': {'Z': 36, 'A': 83.798, 'density': 3.478e-3},
    'Rb': {'Z': 37, 'A': 85.4678, 'density': 1.532},
    'Sr': {'Z': 38, 'A': 87.62, 'density': 2.540},
    'Y': {'Z': 39, 'A': 88.9059, 'density': 4.469},
    'Zr': {'Z': 40, 'A': 91.224, 'density': 6.506},
    'Nb': {'Z': 41, 'A': 92.9064, 'density': 8.570},
    'Mo': {'Z': 42, 'A': 95.96, 'density': 10.220},
    'Tc': {'Z': 43, 'A': 98.0, 'density': 11.500},
    'Ru': {'Z': 44, 'A': 101.07, 'density': 12.410},
    'Rh': {'Z': 45, 'A': 102.906, 'density': 12.410},
    'Pd': {'Z': 46, 'A': 106.42, 'density': 12.020},
    'Ag': {'Z': 47, 'A': 107.868, 'density': 10.500},
    'Cd': {'Z': 48, 'A': 112.411, 'density': 8.650},
    'In': {'Z': 49, 'A': 114.818, 'density': 7.310},
    'Sn': {'Z': 50, 'A': 118.710, 'density': 7.310},
    'Sb': {'Z': 51, 'A': 121.760, 'density': 6.691},
    'Te': {'Z': 52, 'A': 127.60, 'density': 6.240},
    'I': {'Z': 53, 'A': 126.904, 'density': 4.930},
    'Xe': {'Z': 54, 'A': 131.293, 'density': 5.485e-3},
    'Cs': {'Z': 55, 'A': 132.905, 'density': 1.873},
    'Ba': {'Z': 56, 'A': 137.327, 'density': 3.500},
    'La': {'Z': 57, 'A': 138.905, 'density': 6.154},
    'Ce': {'Z': 58, 'A': 140.116, 'density': 6.657},
    'Pr': {'Z': 59, 'A': 140.908, 'density': 6.710},
    'Nd': {'Z': 60, 'A': 144.242, 'density': 6.900},
    'Pm': {'Z': 61, 'A': 145.0, 'density': 7.220},
    'Sm': {'Z': 62, 'A': 150.36, 'density': 7.520},
    'Eu': {'Z': 63, 'A': 151.964, 'density': 5.243},
    'Gd': {'Z': 64, 'A': 157.25, 'density': 7.900},
    'Tb': {'Z': 65, 'A': 158.925, 'density': 8.229},
    'Dy': {'Z': 66, 'A': 162.500, 'density': 8.550},
    'Ho': {'Z': 67, 'A': 164.930, 'density': 8.795},
    'Er': {'Z': 68, 'A': 167.259, 'density': 9.066},
    'Tm': {'Z': 69, 'A': 168.934, 'density': 9.321},
    'Yb': {'Z': 70, 'A': 173.054, 'density': 6.965},
    'Lu': {'Z': 71, 'A': 174.967, 'density': 9.840},
    'Hf': {'Z': 72, 'A': 178.49, 'density': 13.310},
    'Ta': {'Z': 73, 'A': 180.948, 'density': 16.654},
    'W': {'Z': 74, 'A': 183.84, 'density': 19.300},
    'Re': {'Z': 75, 'A': 186.207, 'density': 21.020},
    'Os': {'Z': 76, 'A': 190.23, 'density': 22.570},
    'Ir': {'Z': 77, 'A': 192.217, 'density': 22.420},
    'Pt': {'Z': 78, 'A': 195.084, 'density': 21.450},
    'Au': {'Z': 79, 'A': 196.967, 'density': 19.320},
    'Hg': {'Z': 80, 'A': 200.59, 'density': 13.546},  # Liquid
    'Tl': {'Z': 81, 'A': 204.383, 'density': 11.720},
    'Pb': {'Z': 82, 'A': 207.2, 'density': 11.350},
    'Bi': {'Z': 83, 'A': 208.980, 'density': 9.747},
    'Po': {'Z': 84, 'A': 209.0, 'density': 9.320},
    'At': {'Z': 85, 'A': 210.0, 'density': 7.000},  # Estimated
    'Rn': {'Z': 86, 'A': 222.0, 'density': 9.066e-3},
    'Fr': {'Z': 87, 'A': 223.0, 'density': 1.870},  # Estimated
    'Ra': {'Z': 88, 'A': 226.0, 'density': 5.000},
    'Ac': {'Z': 89, 'A': 227.0, 'density': 10.070},
    'Th': {'Z': 90, 'A': 232.038, 'density': 11.720},
    'Pa': {'Z': 91, 'A': 231.036, 'density': 15.370},
    'U': {'Z': 92, 'A': 238.029, 'density': 18.950},
    'Np': {'Z': 93, 'A': 237.0, 'density': 20.250},
    'Pu': {'Z': 94, 'A': 244.0, 'density': 19.840},
    'Am': {'Z': 95, 'A': 243.0, 'density': 13.670},
    'Cm': {'Z': 96, 'A': 247.0, 'density': 13.510},
}


# =============================================================================
# NIST Standard Compounds (Table 2)
# =============================================================================

NIST_COMPOUNDS = {
    'A-150 Tissue-Equivalent Plastic': Material(
        name='A-150 Tissue-Equivalent Plastic',
        density=1.127,
        composition={'H': 0.101327, 'C': 0.775501, 'N': 0.035057,
                    'O': 0.052316, 'F': 0.017422, 'Ca': 0.018378},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Adipose Tissue (ICRP)': Material(
        name='Adipose Tissue (ICRP)',
        density=0.950,
        composition={'H': 0.114, 'C': 0.598, 'N': 0.007, 'O': 0.278,
                    'Na': 0.001, 'S': 0.001, 'Cl': 0.001},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Air, Dry (near sea level)': Material(
        name='Air, Dry (near sea level)',
        density=0.001205,
        composition={'C': 0.000124, 'N': 0.755268, 'O': 0.231781, 'Ar': 0.012827},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Bakelite': Material(
        name='Bakelite',
        density=1.250,
        composition={'H': 0.057441, 'C': 0.774591, 'O': 0.167968},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Blood (ICRP)': Material(
        name='Blood (ICRP)',
        density=1.060,
        composition={'H': 0.102, 'C': 0.110, 'N': 0.033, 'O': 0.745,
                    'Na': 0.001, 'P': 0.001, 'S': 0.002, 'Cl': 0.003,
                    'K': 0.002, 'Fe': 0.001},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Bone, Compact (ICRU)': Material(
        name='Bone, Compact (ICRU)',
        density=1.850,
        composition={'H': 0.064, 'C': 0.278, 'N': 0.027, 'O': 0.410,
                    'Mg': 0.002, 'P': 0.070, 'S': 0.002, 'Ca': 0.147},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Bone, Cortical (ICRP)': Material(
        name='Bone, Cortical (ICRP)',
        density=1.920,
        composition={'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435,
                    'Na': 0.001, 'Mg': 0.002, 'P': 0.103, 'S': 0.003,
                    'Ca': 0.225},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Brain (ICRP)': Material(
        name='Brain (ICRP)',
        density=1.040,
        composition={'H': 0.107, 'C': 0.145, 'N': 0.022, 'O': 0.712,
                    'Na': 0.002, 'P': 0.004, 'S': 0.002, 'Cl': 0.003,
                    'K': 0.003},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Calcium Fluoride': Material(
        name='Calcium Fluoride',
        density=3.180,
        composition={'Ca': 0.513411, 'F': 0.486589},
        formula='CaF2',
        source='NIST XCOM Table 2'
    ),
    'Cesium Iodide': Material(
        name='Cesium Iodide',
        density=4.510,
        composition={'Cs': 0.511549, 'I': 0.488451},
        formula='CsI',
        source='NIST XCOM Table 2'
    ),
    'Concrete, Ordinary': Material(
        name='Concrete, Ordinary',
        density=2.300,
        composition={'H': 0.010, 'C': 0.001, 'O': 0.529107, 'Na': 0.016,
                    'Mg': 0.002, 'Al': 0.033872, 'Si': 0.337021,
                    'K': 0.013, 'Ca': 0.044, 'Fe': 0.014},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Concrete, Barite': Material(
        name='Concrete, Barite',
        density=3.350,
        composition={'H': 0.003585, 'O': 0.311622, 'Mg': 0.001195,
                    'Al': 0.004183, 'Si': 0.010457, 'S': 0.107858,
                    'Ca': 0.050194, 'Fe': 0.047505, 'Ba': 0.463400},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Glass, Borosilicate': Material(
        name='Glass, Borosilicate',
        density=2.230,
        composition={'B': 0.040066, 'O': 0.539559, 'Na': 0.028191,
                    'Al': 0.011644, 'Si': 0.377220, 'K': 0.003321},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Glass, Lead': Material(
        name='Glass, Lead',
        density=6.220,
        composition={'O': 0.156453, 'Si': 0.080866, 'Ti': 0.008092,
                    'As': 0.002651, 'Pb': 0.751938},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Lung (ICRP)': Material(
        name='Lung (ICRP)',
        density=1.050,
        composition={'H': 0.103, 'C': 0.105, 'N': 0.031, 'O': 0.749,
                    'Na': 0.002, 'P': 0.002, 'S': 0.003, 'Cl': 0.003,
                    'K': 0.002},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Muscle, Skeletal (ICRP)': Material(
        name='Muscle, Skeletal (ICRP)',
        density=1.050,
        composition={'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.710,
                    'Na': 0.001, 'P': 0.002, 'S': 0.003, 'Cl': 0.001,
                    'K': 0.004},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Nylon, Type 6/6': Material(
        name='Nylon, Type 6/6',
        density=1.140,
        composition={'H': 0.097976, 'C': 0.636856, 'N': 0.123779, 'O': 0.141389},
        formula='C12H22N2O2',
        source='NIST XCOM Table 2'
    ),
    'Paraffin Wax': Material(
        name='Paraffin Wax',
        density=0.930,
        composition={'H': 0.148605, 'C': 0.851395},
        formula='C25H52',
        source='NIST XCOM Table 2'
    ),
    'Photographic Emulsion': Material(
        name='Photographic Emulsion',
        density=3.815,
        composition={'H': 0.014, 'C': 0.072, 'N': 0.019, 'O': 0.066,
                    'S': 0.001, 'Br': 0.349, 'Ag': 0.474, 'I': 0.005},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Plastic Scintillator (Vinyltoluene)': Material(
        name='Plastic Scintillator (Vinyltoluene)',
        density=1.032,
        composition={'H': 0.085, 'C': 0.915},
        formula='C9H10',
        source='NIST XCOM Table 2'
    ),
    'Polyethylene': Material(
        name='Polyethylene',
        density=0.940,
        composition={'H': 0.143711, 'C': 0.856289},
        formula='(C2H4)n',
        source='NIST XCOM Table 2'
    ),
    'Polyethylene Terephthalate (Mylar)': Material(
        name='Polyethylene Terephthalate (Mylar)',
        density=1.400,
        composition={'H': 0.041959, 'C': 0.625017, 'O': 0.333025},
        formula='C10H8O4',
        source='NIST XCOM Table 2'
    ),
    'Polymethyl Methacrylate (PMMA)': Material(
        name='Polymethyl Methacrylate (PMMA)',
        density=1.190,
        composition={'H': 0.080538, 'C': 0.599848, 'O': 0.319614},
        formula='C5H8O2',
        source='NIST XCOM Table 2'
    ),
    'Polystyrene': Material(
        name='Polystyrene',
        density=1.060,
        composition={'H': 0.077418, 'C': 0.922582},
        formula='(C8H8)n',
        source='NIST XCOM Table 2'
    ),
    'PTFE (Teflon)': Material(
        name='PTFE (Teflon)',
        density=2.200,
        composition={'C': 0.240183, 'F': 0.759817},
        formula='(C2F4)n',
        source='NIST XCOM Table 2'
    ),
    'PVC': Material(
        name='PVC',
        density=1.300,
        composition={'H': 0.048382, 'C': 0.384361, 'Cl': 0.567257},
        formula='(C2H3Cl)n',
        source='NIST XCOM Table 2'
    ),
    'Silicon Dioxide': Material(
        name='Silicon Dioxide',
        density=2.200,
        composition={'O': 0.532565, 'Si': 0.467435},
        formula='SiO2',
        source='NIST XCOM Table 2'
    ),
    'Sodium Iodide': Material(
        name='Sodium Iodide',
        density=3.667,
        composition={'Na': 0.153373, 'I': 0.846627},
        formula='NaI',
        source='NIST XCOM Table 2'
    ),
    'Stainless Steel (304)': Material(
        name='Stainless Steel (304)',
        density=8.000,
        composition={'C': 0.0008, 'Si': 0.01, 'Cr': 0.19, 'Mn': 0.02,
                    'Fe': 0.6892, 'Ni': 0.09},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Tissue, Soft (ICRP)': Material(
        name='Tissue, Soft (ICRP)',
        density=1.000,
        composition={'H': 0.104472, 'C': 0.232190, 'N': 0.024880,
                    'O': 0.630238, 'Na': 0.001130, 'Mg': 0.000130,
                    'P': 0.001330, 'S': 0.001990, 'Cl': 0.001340,
                    'K': 0.001990, 'Ca': 0.000230, 'Fe': 0.000050,
                    'Zn': 0.000030},
        formula='',
        source='NIST XCOM Table 2'
    ),
    'Water': Material(
        name='Water',
        density=1.000,
        composition={'H': 0.111894, 'O': 0.888106},
        formula='H2O',
        source='NIST XCOM Table 2'
    ),
}


# =============================================================================
# Additional Materials from PNNL Compendium
# =============================================================================

COMPENDIUM_MATERIALS = {
    'Aluminum, Alloy 6061': Material(
        name='Aluminum, Alloy 6061',
        density=2.700,
        composition={'Mg': 0.01, 'Al': 0.9685, 'Si': 0.006,
                    'Ti': 0.0015, 'Cr': 0.002, 'Mn': 0.0015,
                    'Fe': 0.007, 'Cu': 0.004},
        formula='',
        source='PNNL Compendium'
    ),
    'Steel, Carbon': Material(
        name='Steel, Carbon',
        density=7.820,
        composition={'C': 0.005, 'Fe': 0.995},
        formula='',
        source='PNNL Compendium'
    ),
    'Steel, Stainless 316': Material(
        name='Steel, Stainless 316',
        density=8.000,
        composition={'C': 0.0008, 'Si': 0.01, 'Cr': 0.17, 'Mn': 0.02,
                    'Fe': 0.6492, 'Ni': 0.12, 'Mo': 0.03},
        formula='',
        source='PNNL Compendium'
    ),
    'Tungsten Carbide': Material(
        name='Tungsten Carbide',
        density=15.630,
        composition={'C': 0.061, 'W': 0.939},
        formula='WC',
        source='PNNL Compendium'
    ),
    'Brass': Material(
        name='Brass',
        density=8.070,
        composition={'Cu': 0.70, 'Zn': 0.30},
        formula='',
        source='PNNL Compendium'
    ),
    'Bronze': Material(
        name='Bronze',
        density=8.800,
        composition={'Cu': 0.88, 'Sn': 0.12},
        formula='',
        source='PNNL Compendium'
    ),
    'Kapton': Material(
        name='Kapton',
        density=1.420,
        composition={'H': 0.026362, 'C': 0.691133, 'N': 0.073270, 'O': 0.209235},
        formula='C22H10N2O5',
        source='PNNL Compendium'
    ),
    'Salt (NaCl)': Material(
        name='Salt (NaCl)',
        density=2.170,
        composition={'Na': 0.393372, 'Cl': 0.606628},
        formula='NaCl',
        source='PNNL Compendium'
    ),
    'Soil (Average)': Material(
        name='Soil (Average)',
        density=1.520,
        composition={'H': 0.021, 'C': 0.016, 'O': 0.577, 'Al': 0.050,
                    'Si': 0.271, 'K': 0.013, 'Ca': 0.041, 'Fe': 0.011},
        formula='',
        source='PNNL Compendium'
    ),
    'Rock (Average)': Material(
        name='Rock (Average)',
        density=2.650,
        composition={'O': 0.474, 'Na': 0.024, 'Mg': 0.019, 'Al': 0.077,
                    'Si': 0.282, 'K': 0.022, 'Ca': 0.036, 'Fe': 0.043},
        formula='',
        source='PNNL Compendium'
    ),
    'EUROFER97': Material(
        name='EUROFER97',
        density=7.798,
        composition={'C': 0.0011, 'V': 0.002, 'Cr': 0.09, 'Mn': 0.004,
                    'Fe': 0.8899, 'W': 0.011, 'Ta': 0.001, 'N': 0.0003},
        formula='',
        source='PNNL Compendium'
    ),
    'F82H': Material(
        name='F82H',
        density=7.890,
        composition={'C': 0.001, 'V': 0.002, 'Cr': 0.08, 'Mn': 0.005,
                    'Fe': 0.8995, 'W': 0.02, 'Ta': 0.0004, 'B': 0.00001},
        formula='',
        source='PNNL Compendium'
    ),
}


# =============================================================================
# API Functions
# =============================================================================


def get_material(name: str) -> Material:
    """
    Get material definition by name.
    
    Parameters
    ----------
    name : str
        Material name (case-insensitive for common names)
    
    Returns
    -------
    Material
        Material definition with composition
    
    Raises
    ------
    ValueError
        If material not found
    
    Examples
    --------
    >>> mat = get_material('Water')
    >>> mat.density
    1.0
    >>> mat.composition
    {'H': 0.111894, 'O': 0.888106}
    """
    # Check NIST compounds first
    if name in NIST_COMPOUNDS:
        return NIST_COMPOUNDS[name]
    
    # Check Compendium
    if name in COMPENDIUM_MATERIALS:
        return COMPENDIUM_MATERIALS[name]
    
    # Try case-insensitive match
    name_lower = name.lower()
    for mat_name, mat in NIST_COMPOUNDS.items():
        if mat_name.lower() == name_lower:
            return mat
    for mat_name, mat in COMPENDIUM_MATERIALS.items():
        if mat_name.lower() == name_lower:
            return mat
    
    # Check if it's an element
    if name in NIST_ELEMENTS:
        elem = NIST_ELEMENTS[name]
        return Material(
            name=name,
            density=elem['density'],
            composition={name: 1.0},
            formula=name,
            source='NIST XCOM Table 1'
        )
    
    available = list_materials()
    raise ValueError(f"Material '{name}' not found. Use list_materials() to see available.")


def list_materials() -> List[str]:
    """List all available material names."""
    names = list(NIST_COMPOUNDS.keys()) + list(COMPENDIUM_MATERIALS.keys())
    return sorted(names)


def list_elements() -> List[str]:
    """List all element symbols."""
    return sorted(NIST_ELEMENTS.keys())


def get_element_data(symbol: str) -> Dict:
    """
    Get element data by symbol.
    
    Parameters
    ----------
    symbol : str
        Element symbol (e.g., 'Fe', 'U')
    
    Returns
    -------
    dict
        Element data with Z, A, density
    """
    if symbol not in NIST_ELEMENTS:
        raise ValueError(f"Element '{symbol}' not found.")
    return NIST_ELEMENTS[symbol].copy()


def search_materials(query: str) -> List[str]:
    """
    Search materials by partial name match.
    
    Parameters
    ----------
    query : str
        Search string (case-insensitive)
    
    Returns
    -------
    list
        Matching material names
    """
    query_lower = query.lower()
    matches = []
    
    for name in NIST_COMPOUNDS:
        if query_lower in name.lower():
            matches.append(name)
    
    for name in COMPENDIUM_MATERIALS:
        if query_lower in name.lower():
            matches.append(name)
    
    return sorted(matches)


def effective_atomic_number(composition: Dict[str, float]) -> float:
    """
    Calculate effective atomic number for a compound.
    
    Uses the power-law approximation for photon interactions:
    Z_eff = (Σ wᵢ Zᵢ^3.5)^(1/3.5)
    
    Parameters
    ----------
    composition : dict
        Element weight fractions {symbol: fraction}
    
    Returns
    -------
    float
        Effective atomic number
    """
    total = 0.0
    exp = 3.5  # Exponent for photoelectric effect
    
    for symbol, weight in composition.items():
        if symbol in NIST_ELEMENTS:
            Z = NIST_ELEMENTS[symbol]['Z']
            total += weight * (Z ** exp)
    
    return total ** (1.0 / exp)
