"""Diagnostic: verify customer_service_emails.csv can be reliably read."""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

CSV_PATH = Path(__file__).parent / "customer_service_emails.csv"


def guess_type(series: pd.Series) -> str:
    sample = series.dropna().head(20)
    if sample.empty:
        return "unknown (all null)"
    if pd.to_numeric(sample, errors="coerce").notna().all():
        return "numeric"
    if pd.to_datetime(sample, errors="coerce").notna().all():
        return "date"
    return "string"


def main() -> None:
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {CSV_PATH} is empty")
        sys.exit(1)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # subject is expected to show 0 nulls: missing subjects are encoded as the
    # literal string "(no subject)" rather than an empty/null value (e.g. HB-10273)
    print("\nNull counts per column:")
    for col, count in df.isnull().sum().items():
        print(f"  {col}: {count}")

    print("\nInferred type per column (guess from sample, not validated):")
    for col in df.columns:
        print(f"  {col}: {guess_type(df[col])}")


if __name__ == "__main__":
    main()
