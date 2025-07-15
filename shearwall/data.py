from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class ThicknessData:
    nominal_thickness: float
    no_of_plies: int
    Ba_0: int
    Ba_90: int
    Bv: int

# Thickness table
THICKNESS_TABLE: List[ThicknessData] = [
    ThicknessData(7.5, 3, 55000, 24000, 3400),
    ThicknessData(9.5, 3, 55000, 28000, 4300),
    ThicknessData(12.5, 3, 81000, 39000, 5700),
    ThicknessData(15.5, 4, 59000, 75000, 7100),
    ThicknessData(18.5, 5, 83000, 69000, 8600),
    ThicknessData(20.5, 5, 100000, 89000, 9500),
    ThicknessData(22.5, 6, 130000, 69000, 10000),
    ThicknessData(25.5, 7, 120000, 98000, 12000),
]

# Panel BV values
BV_VALUE_DF = pd.DataFrame([
    {"Panel Type": "DFP", "Thickness": 9.5, "BV": 5500},
    {"Panel Type": "DFP", "Thickness": 12.5, "BV": 6900},
    {"Panel Type": "DFP", "Thickness": 15.5, "BV": 8400},
    {"Panel Type": "DFP", "Thickness": 18.5, "BV": 9800},
    {"Panel Type": "CSP", "Thickness": 9.5, "BV": 4300},
    {"Panel Type": "CSP", "Thickness": 12.5, "BV": 5700},
    {"Panel Type": "CSP", "Thickness": 15.5, "BV": 7100},
    {"Panel Type": "CSP", "Thickness": 18.5, "BV": 8600},
    {"Panel Type": "OSB", "Thickness": 9.5, "BV": 10000},
    {"Panel Type": "OSB", "Thickness": 11.0, "BV": 11000},
    {"Panel Type": "OSB", "Thickness": 12.0, "BV": 11000},
    {"Panel Type": "OSB", "Thickness": 15.0, "BV": 11000},
    {"Panel Type": "OSB", "Thickness": 18.0, "BV": 12000},
])

# Elasticity values
E_VALUE_DF = pd.DataFrame([
    {"Species": "D.Fir-L", "Grade": "S.S.", "Epar": 12500, "Eperp": 625},
    {"Species": "D.Fir-L", "Grade": "No.1/No.2", "Epar": 11000, "Eperp": 550},
    {"Species": "D.Fir-L", "Grade": "No.3/Stud", "Epar": 10000, "Eperp": 500},
    {"Species": "Hem-Fir", "Grade": "S.S.", "Epar": 12000, "Eperp": 600},
    {"Species": "Hem-Fir", "Grade": "No.1/No.2", "Epar": 11000, "Eperp": 550},
    {"Species": "Hem-Fir", "Grade": "No.3/Stud", "Epar": 10000, "Eperp": 500},
    {"Species": "S-P-F", "Grade": "S.S.", "Epar": 10500, "Eperp": 525},
    {"Species": "S-P-F", "Grade": "No.1/No.2", "Epar": 9500, "Eperp": 475},
    {"Species": "S-P-F", "Grade": "No.3/Stud", "Epar": 9000, "Eperp": 450},
    {"Species": "Northern", "Grade": "S.S.", "Epar": 7500, "Eperp": 375},
    {"Species": "Northern", "Grade": "No.1/No.2", "Epar": 7000, "Eperp": 350},
    {"Species": "Northern", "Grade": "No.3/Stud", "Epar": 6500, "Eperp": 325},
])

# Holddown and rod data
HOLDDOWN_DF = pd.DataFrame({
    'Model No.': ['ATUD6-2', 'ATUD9', 'ATUD9-2', 'ATUD9-3', 'ATUD14', 'TUD10',
                'RTUD4', 'RTUD5', 'RTUD6',
                'CTUD55', 'CTUD65', 'CTUD66', 'CTUD75', 'CTUD76', 'CTUD77',
                'CTUD87', 'CTUD88', 'CTUD97', 'CTUD98', 'CTUD99'],
    'Diameter (in.)': [0.750, 1.125, 1.125, 1.125, 1.750, 1.250,
                    0.500, 0.625, 0.750,
                    0.625, 0.750, 0.750, 0.875, 0.875, 0.875,
                    1.000, 1.000, 1.125, 1.125, 1.125],
    'Seating Increment (DR) (mm)': [0.102, 0.051, 0.051, 0.051, 0.127, 0.025,
                                    1.219, 1.422, 1.448,
                                    0.102, 0.076, 0.076, 0.076, 0.076, 0.076,
                                    0.076, 0.076, 0.076, 0.076, 0.076],
    'Deflection at Factored Resistance (DF) (mm)': [0.635, 0.356, 1.118, 1.016,
                                                    0.432, 0.838,
                                                    0.305, 0.178, 0.279,
                                                    0.203, 1.778, 1.778, 1.778,
                                                    1.778, 1.778,
                                                    0.965, 0.965, 0.965, 0.965,
                                                    0.965]
})

TR_DATA_DF = pd.DataFrame({
    'Model no.': ['SR4', 'SR5', 'SR6', 'SR7', 'SR8', 'SR9', 'SR10',
                'SR4H', 'SR5H', 'SR6H', 'SR7H', 'SR8H', 'SR9H',
                'SR10H', 'SR9H150', 'SR10H150'],
    'Rod diameter': [0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25,
                    0.5, 0.625, 0.75, 0.875, 1, 1.125,
                    1.25, 1.125, 1.25],
    'Tr (kN)': [26.33, 42.45, 62.61, 85.79, 112.12, 141.96, 175.35,
                56.33, 90.6, 133.9, 183.54, 239.87, 303.74,
                375.14, 382.36, 485.59]
})

ROD_DATA = [
    {"Rod Diameter": "1/2\"", "Strong Rod Standard": "SR4", "Strong Rod High Strength": "SR4H", "Ag": 98.7, "Ane": 91.6, "Tr Standard": 26, "Tr High": 56},
    {"Rod Diameter": "5/8\"", "Strong Rod Standard": "SR5", "Strong Rod High Strength": "SR5H", "Ag": 158.7, "Ane": 145.8, "Tr Standard": 43, "Tr High": 91},
    {"Rod Diameter": "3/4\"", "Strong Rod Standard": "SR6", "Strong Rod High Strength": "SR6H", "Ag": 234.2, "Ane": 215.5, "Tr Standard": 63, "Tr High": 134},
    {"Rod Diameter": "7/8\"", "Strong Rod Standard": "SR7", "Strong Rod High Strength": "SR7H", "Ag": 321.3, "Ane": 298.1, "Tr Standard": 86, "Tr High": 184},
    {"Rod Diameter": "1\"", "Strong Rod Standard": "SR8", "Strong Rod High Strength": "SR8H", "Ag": 419.4, "Ane": 391.0, "Tr Standard": 112, "Tr High": 240},
    {"Rod Diameter": "1 1/8\"", "Strong Rod Standard": "SR9", "Strong Rod High Strength": "SR9H", "Ag": 531.6, "Ane": 492.3, "Tr Standard": 142, "Tr High": 304},
    {"Rod Diameter": "1 1/4\"", "Strong Rod Standard": "SR10", "Strong Rod High Strength": "SR10H", "Ag": 670.3, "Ane": 625.2, "Tr Standard": 179, "Tr High": 383},
    {"Rod Diameter": "1 3/8\"", "Strong Rod Standard": "SR11", "Strong Rod High Strength": "SR11H", "Ag": 811.0, "Ane": 745.2, "Tr Standard": 217, "Tr High": 464},
    {"Rod Diameter": "1 1/2\"", "Strong Rod Standard": "SR12", "Strong Rod High Strength": "SR12H", "Ag": 981.9, "Ane": 906.4, "Tr Standard": 262, "Tr High": 561},
    {"Rod Diameter": "1 3/4\"", "Strong Rod Standard": "SR14", "Strong Rod High Strength": "SR14H", "Ag": 1332.9, "Ane": 1225.8, "Tr Standard": 356, "Tr High": 762},
    {"Rod Diameter": "2\"", "Strong Rod Standard": "SR16", "Strong Rod High Strength": "SR16H", "Ag": 1810.3, "Ane": 1611.6, "Tr Standard": 483, "Tr High": 1033}
]

SHEAR_CLIP_DATA_TABLE = [
    {"Stud Species": "D.Fir-L", "A35": 4.25, "LTP4": 3.63, "LTP5": 3.85, "LS70": 4.60, "LS90": 5.52},
    {"Stud Species": "Hem-Fir", "A35": 3.00, "LTP4": 2.58, "LTP5": 2.74, "LS70": 3.58, "LS90": 5.00},
    {"Stud Species": "S-P-F", "A35": 3.00, "LTP4": 2.58, "LTP5": 2.74, "LS70": 3.58, "LS90": 5.00},
    {"Stud Species": "Northern", "A35": 2.55, "LTP4": 2.19, "LTP5": 2.33, "LS70": 3.04, "LS90": 4.25},
]

BEARING_PLATE_DATA = [
    {"Model No.": "BPRTUD3-4B", "D.Fir-L (lb.)": 6120, "D.Fir-L (kN)": 27.26, "S-P-F (lb.)": 6120, "S-P-F (kN)": 27.26},
    {"Model No.": "BPRTUD5-6A", "D.Fir-L (lb.)": 7060, "D.Fir-L (kN)": 31.45, "S-P-F (lb.)": 7060, "S-P-F (kN)": 31.45},
    {"Model No.": "BPRTUD5-6B", "D.Fir-L (lb.)": 18110, "D.Fir-L (kN)": 80.67, "S-P-F (lb.)": 13705, "S-P-F (kN)": 61.05},
    {"Model No.": "BPRTUD5-6C", "D.Fir-L (lb.)": 23400, "D.Fir-L (kN)": 104.23, "S-P-F (lb.)": 17705, "S-P-F (kN)": 78.86},
    {"Model No.": "BPRTUD5-8", "D.Fir-L (lb.)": 5200, "D.Fir-L (kN)": 23.13, "S-P-F (lb.)": 5200, "S-P-F (kN)": 23.13},
    {"Model No.": "BPRTUD7-8A", "D.Fir-L (lb.)": 17555, "D.Fir-L (kN)": 78.09, "S-P-F (lb.)": 13285, "S-P-F (kN)": 59.10},
    {"Model No.": "BPRTUD7-8B", "D.Fir-L (lb.)": 18740, "D.Fir-L (kN)": 83.36, "S-P-F (lb.)": 18740, "S-P-F (kN)": 83.36},
    {"Model No.": "BPRTUD7-8C", "D.Fir-L (lb.)": 30175, "D.Fir-L (kN)": 134.23, "S-P-F (lb.)": 22835, "S-P-F (kN)": 101.58},
]
