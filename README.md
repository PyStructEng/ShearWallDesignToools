# Shear Wall Design Tools

This project provides Python utilities that integrate with Excel to perform shear wall structural analysis. Calculations of shear capacity, deflection and bearing plate forces are automated in Python while results are written back to Excel workbooks.

## Installation

1. Install **Python 3.8+**.
2. Install the required packages:

```bash
pip install numpy pandas matplotlib shapely xlwings rich pycba
```

`xlwings` requires Microsoft Excel (Windows or macOS) and the xlwings add-in.

## Running the Excel Analysis

1. Open `Wood_lateral_Spreadsheet_R8.xlsm` or your workbook based on it.
2. Enable macros and make sure the xlwings add-in is installed.
3. Trigger the analysis macro (for example a button linked to `indv_shear_wall_analysis`). Excel calls into Python using `xw.Book.caller()` as shown below:

```python
# excerpt from shearwallanalysis.py
def indv_shear_wall_analysis():
    file_name = xw.Book.caller().fullname
    wb = xw.Book(file_name)
    ws = wb.sheets.active
```

The function reads input data from the worksheet, performs the computations, and writes results back to the workbook.

## Usage Example

Running the macro fills the deflection and shear clip tables automatically. You can also run the function from a Python shell if xlwings is set up:

```python
import xlwings as xw
import shearwallanalysis

wb = xw.Book("Wood_lateral_Spreadsheet_R8.xlsm")
shearwallanalysis.indv_shear_wall_analysis()
```

Optionally place screenshots of the workbook or output charts in a `docs/` folder and reference them here.
