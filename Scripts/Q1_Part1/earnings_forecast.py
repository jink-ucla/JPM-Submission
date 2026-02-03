"""
Earnings Forecasting using Balance Sheet Model
-----------------------------------------------
Use the trained balance sheet model to forecast company earnings.

Questions addressed: Part 1, Q6

Key relationships:
- Net Income is linked to balance sheet through retained earnings
- Retained Earnings(t) = Retained Earnings(t-1) + Net Income(t) - Dividends(t)
- Can infer earnings from balance sheet forecasts
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from tensorflow_model import BalanceSheetForecaster, create_sequences
from data_collection import FinancialDataCollector


class EarningsForecaster:
    """Forecast earnings using balance sheet predictions."""

    def __init__(self, balance_sheet_model: BalanceSheetForecaster):
        """
        Initialize earnings forecaster.

        Args:
            balance_sheet_model: Trained balance sheet forecasting model
        """
        self.bs_model = balance_sheet_model

    def extract_earnings_from_balance_sheet(self,
                                           balance_sheet: pd.DataFrame,
                                           method: str = 'retained_earnings') -> pd.Series:
        """
        Extract earnings information from balance sheet.

        Args:
            balance_sheet: Balance sheet DataFrame
            method: Method to use ('retained_earnings' or 'equity_change')

        Returns:
            Series of earnings estimates
        """
        if method == 'retained_earnings':
            # Net Income = Delta(Retained Earnings) + Dividends
            # If we don't have dividends, approximate as just delta
            if 'Retained Earnings' in balance_sheet.columns:
                retained_earnings = balance_sheet['Retained Earnings']
                net_income = retained_earnings.diff()
                return net_income

        elif method == 'equity_change':
            # Net Income contributes to equity change
            # Delta(Equity) = Net Income - Dividends + New Equity Issued
            if 'Stockholders Equity' in balance_sheet.columns:
                equity = balance_sheet['Stockholders Equity']
                equity_change = equity.diff()
                # This is approximate without knowing dividends and equity issuance
                return equity_change

        return pd.Series(dtype=float)

    def forecast_earnings(self,
                         historical_data: pd.DataFrame,
                         forecast_periods: int = 4) -> Dict:
        """
        Forecast earnings for future periods.

        Args:
            historical_data: Historical financial data
            forecast_periods: Number of periods to forecast

        Returns:
            Dictionary with earnings forecasts and confidence intervals
        """
        # Prepare data for balance sheet model
        balance_sheet_cols = [col for col in historical_data.columns if any(
            keyword in col for keyword in
            ['Assets', 'Liabilities', 'Equity', 'Cash', 'Debt', 'Inventory']
        )]

        bs_data = historical_data[balance_sheet_cols].dropna()

        # Normalize
        normalized_data = self.bs_model.preprocess_data(bs_data)

        # Forecast balance sheet
        bs_forecast = self.bs_model.forecast(normalized_data, steps=forecast_periods)

        # Denormalize
        bs_forecast_denorm = self.bs_model.scaler.inverse_transform(bs_forecast)
        bs_forecast_df = pd.DataFrame(bs_forecast_denorm, columns=balance_sheet_cols)

        # Extract earnings from forecasted balance sheet
        if 'Retained Earnings' in bs_forecast_df.columns:
            # Append last historical value to calculate difference
            last_historical = bs_data[['Retained Earnings']].iloc[-1]
            full_re = pd.concat([
                last_historical.to_frame().T,
                bs_forecast_df[['Retained Earnings']]
            ], ignore_index=True)

            earnings_forecast = full_re['Retained Earnings'].diff().dropna()
        else:
            # Fallback: use equity changes
            earnings_forecast = pd.Series([np.nan] * forecast_periods)

        return {
            'earnings_forecast': earnings_forecast.values,
            'balance_sheet_forecast': bs_forecast_df,
            'periods': forecast_periods
        }

    def calculate_earnings_metrics(self,
                                   earnings: pd.Series,
                                   balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate earnings-related financial metrics.

        Args:
            earnings: Net income series
            balance_sheet: Balance sheet DataFrame

        Returns:
            Dictionary of earnings metrics
        """
        metrics = {}

        # ROE (Return on Equity)
        if 'Stockholders Equity' in balance_sheet.columns:
            roe = earnings / balance_sheet['Stockholders Equity']
            metrics['ROE'] = roe

        # ROA (Return on Assets)
        if 'Total Assets' in balance_sheet.columns:
            roa = earnings / balance_sheet['Total Assets']
            metrics['ROA'] = roa

        # Earnings growth rate
        earnings_growth = earnings.pct_change()
        metrics['Earnings_Growth'] = earnings_growth

        # Earnings volatility
        metrics['Earnings_Volatility'] = earnings.std() / earnings.mean() if earnings.mean() != 0 else np.nan

        return metrics


def visualize_earnings_forecast(historical_earnings: pd.Series,
                                forecasted_earnings: np.ndarray,
                                ticker: str,
                                save_path: str = None):
    """
    Visualize earnings forecast.

    Args:
        historical_earnings: Historical earnings data
        forecasted_earnings: Forecasted earnings
        ticker: Stock ticker
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical
    ax.plot(range(len(historical_earnings)), historical_earnings,
            'b-', label='Historical Earnings', linewidth=2)

    # Forecast
    forecast_start = len(historical_earnings)
    forecast_periods = range(forecast_start, forecast_start + len(forecasted_earnings))
    ax.plot(forecast_periods, forecasted_earnings,
            'r--', label='Forecasted Earnings', linewidth=2, marker='o')

    # Add connecting line
    ax.plot([forecast_start - 1, forecast_start],
            [historical_earnings.iloc[-1], forecasted_earnings[0]],
            'g:', linewidth=1)

    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Net Income', fontsize=12)
    ax.set_title(f'Earnings Forecast for {ticker}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    """Main earnings forecasting pipeline."""
    parser = argparse.ArgumentParser(description='Forecast earnings using balance sheet model')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--data-path', default='data', help='Path to data directory')
    parser.add_argument('--model-path', default='models/balance_sheet_model.h5',
                       help='Path to trained model')
    parser.add_argument('--forecast-periods', type=int, default=4,
                       help='Number of periods to forecast')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')

    args = parser.parse_args()

    print(f"Forecasting Earnings for {args.ticker}")
    print("=" * 80)

    # Load data
    data_file = Path(args.data_path) / f"{args.ticker}_combined.csv"
    if not data_file.exists():
        # Collect data if not exists
        print(f"Data file not found. Collecting data for {args.ticker}...")
        collector = FinancialDataCollector(output_dir=args.data_path)
        collector.collect_multiple_companies([args.ticker])

        # Create combined dataset
        combined = collector.create_combined_dataset(args.ticker)
        data_file = Path(args.data_path) / f"{args.ticker}_combined.csv"
        combined.to_csv(data_file)

    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded data: {df.shape}")

    # Initialize balance sheet model
    bs_forecaster = BalanceSheetForecaster(
        sequence_length=4,
        forecast_horizon=1,
        lstm_units=[128, 64],
        dense_units=[64, 32]
    )

    # Load or train model
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"Loading model from {model_path}")
        # Note: You would need to implement model loading
        # For now, we'll train a simple model
        print("Note: Model loading not implemented, training new model...")

    # Train model on balance sheet data
    balance_sheet_cols = [col for col in df.columns if any(
        keyword in col for keyword in
        ['Assets', 'Liabilities', 'Equity', 'Cash', 'Debt']
    )]

    if balance_sheet_cols:
        df_bs = df[balance_sheet_cols].dropna()
        normalized_data = bs_forecaster.preprocess_data(df_bs)

        # Create sequences and train
        X, y = create_sequences(normalized_data, sequence_length=4)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        bs_forecaster.build_model(input_dim=X.shape[-1])
        print("\nTraining model...")
        bs_forecaster.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)

    # Initialize earnings forecaster
    earnings_forecaster = EarningsForecaster(bs_forecaster)

    # Forecast earnings
    print(f"\nForecasting {args.forecast_periods} periods ahead...")
    forecast_results = earnings_forecaster.forecast_earnings(
        df, forecast_periods=args.forecast_periods
    )

    # Display results
    print("\nEarnings Forecast:")
    print("-" * 80)
    for i, earnings in enumerate(forecast_results['earnings_forecast'], 1):
        print(f"  Period {i}: ${earnings:,.2f}")

    # Calculate metrics
    if 'Net Income' in df.columns:
        historical_earnings = df['Net Income'].dropna()
        metrics = earnings_forecaster.calculate_earnings_metrics(
            historical_earnings,
            df[balance_sheet_cols].dropna()
        )

        print("\nHistorical Earnings Metrics:")
        print("-" * 80)
        for metric_name, metric_values in metrics.items():
            if hasattr(metric_values, 'mean'):
                print(f"  {metric_name} (mean): {metric_values.mean():.4f}")

        # Visualize
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        visualize_earnings_forecast(
            historical_earnings,
            forecast_results['earnings_forecast'],
            args.ticker,
            save_path=output_dir / f"{args.ticker}_earnings_forecast.png"
        )

    # Save results
    output_file = output_dir / f"{args.ticker}_earnings_forecast.csv"
    forecast_df = pd.DataFrame({
        'Period': range(1, len(forecast_results['earnings_forecast']) + 1),
        'Forecasted_Earnings': forecast_results['earnings_forecast']
    })
    forecast_df.to_csv(output_file, index=False)
    print(f"\nForecast saved to: {output_file}")


if __name__ == "__main__":
    main()
