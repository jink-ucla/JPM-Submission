"""
Data Collection from Yahoo Finance
-----------------------------------
Fetch income statement and balance sheet data for model training.

Questions addressed: Part 1, Q4

Reference: https://rfachrizal.medium.com/how-to-obtain-financial-statements-from-stocks-using-yfinance-87c432b803b8
"""

import yfinance as yf
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime


class FinancialDataCollector:
    """Collect financial statement data from Yahoo Finance."""

    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch income statement and balance sheet for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

        Returns:
            Dictionary containing income_statement, balance_sheet, and cashflow
        """
        print(f"Fetching financial data for {ticker}...")

        stock = yf.Ticker(ticker)

        # Get financial statements
        financials = {
            'income_statement': stock.financials,  # Annual income statement
            'balance_sheet': stock.balance_sheet,  # Annual balance sheet
            'cashflow': stock.cashflow,            # Annual cash flow statement
            'quarterly_income_statement': stock.quarterly_financials,
            'quarterly_balance_sheet': stock.quarterly_balance_sheet,
            'quarterly_cashflow': stock.quarterly_cashflow,
        }

        # Get company info
        try:
            info = stock.info
            financials['company_info'] = pd.Series(info)
        except Exception as e:
            print(f"Warning: Could not fetch company info: {e}")

        return financials

    def preprocess_balance_sheet(self, balance_sheet: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess balance sheet data for modeling.

        Args:
            balance_sheet: Raw balance sheet data from yfinance

        Returns:
            Preprocessed balance sheet with standardized fields
        """
        # Transpose so dates are rows and accounts are columns
        bs = balance_sheet.T.sort_index()

        # Ensure key fields exist (may vary by company)
        required_fields = [
            'Total Assets',
            'Total Liabilities Net Minority Interest',
            'Stockholders Equity',
            'Current Assets',
            'Current Liabilities',
            'Cash And Cash Equivalents',
            'Total Debt',
        ]

        # Check which fields are available
        available_fields = [f for f in required_fields if f in bs.columns]
        missing_fields = [f for f in required_fields if f not in bs.columns]

        if missing_fields:
            print(f"Warning: Missing fields in balance sheet: {missing_fields}")

        return bs

    def preprocess_income_statement(self, income_stmt: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess income statement data for modeling.

        Args:
            income_stmt: Raw income statement from yfinance

        Returns:
            Preprocessed income statement
        """
        # Transpose so dates are rows and accounts are columns
        is_df = income_stmt.T.sort_index()

        # Ensure key fields exist
        required_fields = [
            'Total Revenue',
            'Cost Of Revenue',
            'Operating Income',
            'Net Income',
            'Interest Expense',
            'Tax Provision',
        ]

        available_fields = [f for f in required_fields if f in is_df.columns]
        missing_fields = [f for f in required_fields if f not in is_df.columns]

        if missing_fields:
            print(f"Warning: Missing fields in income statement: {missing_fields}")

        return is_df

    def verify_accounting_identity(self, balance_sheet: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that Assets = Liabilities + Equity for all periods.

        Args:
            balance_sheet: Balance sheet dataframe

        Returns:
            DataFrame showing verification results
        """
        try:
            assets = balance_sheet['Total Assets']
            liabilities = balance_sheet.get('Total Liabilities Net Minority Interest',
                                           balance_sheet.get('Total Liabilities', 0))
            equity = balance_sheet.get('Stockholders Equity',
                                      balance_sheet.get('Total Equity', 0))

            verification = pd.DataFrame({
                'Assets': assets,
                'Liabilities': liabilities,
                'Equity': equity,
                'Liabilities + Equity': liabilities + equity,
                'Difference': assets - (liabilities + equity),
                'Relative Error (%)': ((assets - (liabilities + equity)) / assets * 100)
            })

            return verification
        except KeyError as e:
            print(f"Error verifying accounting identity: {e}")
            return pd.DataFrame()

    def collect_multiple_companies(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Collect financial data for multiple companies.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to financial data
        """
        all_data = {}

        for ticker in tickers:
            try:
                data = self.fetch_financial_statements(ticker)
                all_data[ticker] = data

                # Save individual company data
                self.save_company_data(ticker, data)

                print(f"[OK] Successfully collected data for {ticker}")
            except Exception as e:
                print(f"[ERROR] Failed to collect data for {ticker}: {e}")

        return all_data

    def save_company_data(self, ticker: str, data: Dict[str, pd.DataFrame]):
        """
        Save company financial data to CSV files.

        Args:
            ticker: Stock ticker
            data: Dictionary of financial statement dataframes
        """
        ticker_dir = self.output_dir / ticker
        ticker_dir.mkdir(exist_ok=True)

        # Save each statement type
        for stmt_type, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = ticker_dir / f"{stmt_type}.csv"
                df.to_csv(filepath)
                print(f"  Saved {stmt_type} to {filepath}")
            elif isinstance(df, pd.Series):
                filepath = ticker_dir / f"{stmt_type}.json"
                df.to_json(filepath, indent=2)
                print(f"  Saved {stmt_type} to {filepath}")

    def create_combined_dataset(self, ticker: str) -> pd.DataFrame:
        """
        Combine balance sheet and income statement into single dataset.

        Args:
            ticker: Stock ticker

        Returns:
            Combined dataframe with all financial metrics
        """
        data = self.fetch_financial_statements(ticker)

        # Preprocess statements
        bs = self.preprocess_balance_sheet(data['balance_sheet'])
        is_df = self.preprocess_income_statement(data['income_statement'])
        cf = data['cashflow'].T.sort_index()

        # Combine on date index
        combined = pd.concat([bs, is_df, cf], axis=1, join='inner')

        # Add derived metrics
        combined['Working_Capital'] = (combined['Current Assets'] -
                                       combined['Current Liabilities'])

        # Add time features
        combined['Year'] = combined.index.year
        combined['Quarter'] = combined.index.quarter

        return combined


def main():
    """Main function to demonstrate data collection."""
    parser = argparse.ArgumentParser(description='Collect financial data from Yahoo Finance')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='Stock tickers to collect data for')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for collected data')

    args = parser.parse_args()

    # Initialize collector
    collector = FinancialDataCollector(output_dir=args.output_dir)

    # Collect data for specified tickers
    print(f"Collecting financial data for: {', '.join(args.tickers)}")
    print("=" * 80)

    all_data = collector.collect_multiple_companies(args.tickers)

    # Demonstrate accounting identity verification
    print("\n" + "=" * 80)
    print("Verifying Accounting Identities")
    print("=" * 80)

    for ticker in args.tickers:
        if ticker in all_data:
            print(f"\n{ticker}:")
            bs = collector.preprocess_balance_sheet(all_data[ticker]['balance_sheet'])
            verification = collector.verify_accounting_identity(bs)
            if not verification.empty:
                print(verification.to_string())

    # Create combined datasets
    print("\n" + "=" * 80)
    print("Creating Combined Datasets")
    print("=" * 80)

    for ticker in args.tickers:
        try:
            combined = collector.create_combined_dataset(ticker)
            output_file = collector.output_dir / f"{ticker}_combined.csv"
            combined.to_csv(output_file)
            print(f"[OK] Saved combined dataset for {ticker}: {output_file}")
            print(f"  Shape: {combined.shape}")
            print(f"  Columns: {len(combined.columns)}")
        except Exception as e:
            print(f"[ERROR] Failed to create combined dataset for {ticker}: {e}")


if __name__ == "__main__":
    main()
