"""
Quick script to combine financial statement CSVs into one dataset.
"""
import pandas as pd
from pathlib import Path
import sys

def combine_financials(ticker: str, data_dir: str = "data", quarterly: bool = True):
    """Combine balance sheet, income statement, and cashflow into one CSV."""
    ticker_path = Path(data_dir) / ticker

    if not ticker_path.exists():
        print(f"[ERROR] Directory not found: {ticker_path}")
        return None

    print(f"Combining {'quarterly' if quarterly else 'annual'} data for {ticker}...")

    # Load the three main statements
    prefix = "quarterly_" if quarterly else ""
    files = {
        'balance_sheet': ticker_path / f"{prefix}balance_sheet.csv",
        'income_statement': ticker_path / f"{prefix}income_statement.csv",
        'cashflow': ticker_path / f"{prefix}cashflow.csv"
    }

    dataframes = {}
    for name, filepath in files.items():
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            # Transpose so dates are rows
            df = df.T
            # Add prefix to column names
            df.columns = [f"{name}_{col}" for col in df.columns]
            dataframes[name] = df
            print(f"  Loaded {name}: {df.shape}")
        else:
            print(f"  [WARNING] File not found: {filepath}")

    if not dataframes:
        print("[ERROR] No data files found!")
        return None

    # Combine all dataframes
    combined = pd.concat(dataframes.values(), axis=1)

    # Save combined dataset
    suffix = "_quarterly" if quarterly else ""
    output_file = ticker_path / f"{ticker}_combined{suffix}.csv"
    combined.to_csv(output_file)

    print(f"[OK] Saved combined dataset: {output_file}")
    print(f"  Shape: {combined.shape}")
    print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")

    return combined

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    quarterly = "--quarterly" in sys.argv or "-q" in sys.argv
    combine_financials(ticker, quarterly=quarterly)
