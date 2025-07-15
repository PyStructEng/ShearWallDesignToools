import argparse
from .excel_io import indv_shear_wall_analysis


def main():
    parser = argparse.ArgumentParser(description="Run shear wall analysis")
    parser.add_argument("workbook", help="Path to Excel workbook")
    args = parser.parse_args()
    indv_shear_wall_analysis(args.workbook)


if __name__ == "__main__":
    main()
