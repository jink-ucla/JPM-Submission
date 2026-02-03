"""
Training and Evaluation Pipeline
---------------------------------
Train the balance sheet forecasting model and evaluate its performance.

Questions addressed: Part 1, Q5

Evaluation criteria:
1. Forecast accuracy (MSE, MAE, MAPE)
2. Accounting identity satisfaction
3. Temporal consistency
4. Out-of-sample performance
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

from tensorflow_model import BalanceSheetForecaster, create_sequences
from data_collection import FinancialDataCollector


class ModelEvaluator:
    """Comprehensive evaluation of balance sheet forecasting models."""

    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate comprehensive forecasting metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Ensure same shape
        y_true = y_true.reshape(y_pred.shape)

        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Directional accuracy (for changes)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true, axis=0))
            pred_direction = np.sign(np.diff(y_pred, axis=0))
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = None

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }

    def validate_accounting_identities(self,
                                      predictions: pd.DataFrame,
                                      tolerance: float = 1e-2) -> dict:
        """
        Validate accounting identities in predictions.

        Args:
            predictions: DataFrame with predicted balance sheet values
            tolerance: Acceptable relative error

        Returns:
            Validation results
        """
        results = {}

        # 1. Fundamental equation: Assets = Liabilities + Equity
        if all(col in predictions.columns for col in ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']):
            assets = predictions['Total Assets']
            liabilities = predictions['Total Liabilities Net Minority Interest']
            equity = predictions['Stockholders Equity']

            violation = assets - (liabilities + equity)
            relative_error = np.abs(violation / assets)

            results['fundamental_equation'] = {
                'mean_absolute_error': np.mean(np.abs(violation)),
                'max_absolute_error': np.max(np.abs(violation)),
                'mean_relative_error': np.mean(relative_error) * 100,
                'max_relative_error': np.max(relative_error) * 100,
                'within_tolerance': np.all(relative_error < tolerance),
                'violations': violation.tolist()
            }

        # 2. Current Ratio consistency
        if all(col in predictions.columns for col in ['Current Assets', 'Current Liabilities']):
            current_ratio = predictions['Current Assets'] / predictions['Current Liabilities']
            results['current_ratio'] = {
                'mean': current_ratio.mean(),
                'std': current_ratio.std(),
                'min': current_ratio.min(),
                'max': current_ratio.max(),
                'values': current_ratio.tolist()
            }

        # 3. Debt to Equity ratio consistency
        if all(col in predictions.columns for col in ['Total Debt', 'Stockholders Equity']):
            debt_to_equity = predictions['Total Debt'] / predictions['Stockholders Equity']
            results['debt_to_equity'] = {
                'mean': debt_to_equity.mean(),
                'std': debt_to_equity.std(),
                'values': debt_to_equity.tolist()
            }

        return results

    def evaluate_temporal_consistency(self, predictions: np.ndarray) -> dict:
        """
        Evaluate temporal consistency of predictions.

        Args:
            predictions: Array of predictions over time

        Returns:
            Temporal consistency metrics
        """
        # Calculate period-to-period changes
        changes = np.diff(predictions, axis=0)
        relative_changes = changes / predictions[:-1]

        # Check for unrealistic jumps
        large_changes = np.abs(relative_changes) > 0.5  # More than 50% change

        return {
            'mean_absolute_change': np.mean(np.abs(changes)),
            'std_change': np.std(changes),
            'mean_relative_change': np.mean(np.abs(relative_changes)) * 100,
            'large_change_count': np.sum(large_changes),
            'large_change_percentage': np.mean(large_changes) * 100
        }

    def cross_validate(self,
                      data: np.ndarray,
                      forecaster: BalanceSheetForecaster,
                      n_splits: int = 5) -> dict:
        """
        Perform time series cross-validation.

        Args:
            data: Financial data
            forecaster: Forecasting model
            n_splits: Number of CV splits

        Returns:
            Cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            print(f"  Fold {fold + 1}/{n_splits}...")

            train_data = data[train_idx]
            test_data = data[test_idx]

            # Create sequences
            X_train, y_train = create_sequences(train_data,
                                                sequence_length=forecaster.sequence_length)
            X_test, y_test = create_sequences(test_data,
                                              sequence_length=forecaster.sequence_length)

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Train
            forecaster.build_model(input_dim=X_train.shape[-1])
            forecaster.train(X_train, y_train, X_test, y_test,
                           epochs=50, batch_size=16)

            # Evaluate
            y_pred = forecaster.model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            cv_scores.append(metrics)

        # Aggregate results
        avg_metrics = {}
        for key in cv_scores[0].keys():
            values = [score[key] for score in cv_scores if score[key] is not None]
            if values:
                avg_metrics[f'mean_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)

        return {
            'cv_scores': cv_scores,
            'average_metrics': avg_metrics
        }


def train_model(ticker: str,
                data_path: str,
                sequence_length: int = 4,
                forecast_horizon: int = 1,
                test_size: float = 0.2) -> dict:
    """
    Train and evaluate a balance sheet forecasting model.

    Args:
        ticker: Stock ticker symbol
        data_path: Path to financial data
        sequence_length: Number of historical periods
        forecast_horizon: Number of periods to forecast
        test_size: Fraction of data for testing

    Returns:
        Training results and metrics
    """
    print(f"\nTraining model for {ticker}")
    print("=" * 80)

    # Load data
    data_file = Path(data_path) / f"{ticker}_combined.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded data: {df.shape}")

    # Select relevant columns (balance sheet items)
    balance_sheet_cols = [col for col in df.columns if any(
        keyword in col for keyword in
        ['Assets', 'Liabilities', 'Equity', 'Cash', 'Debt', 'Inventory', 'Receivable']
    )]

    if not balance_sheet_cols:
        print("Warning: No balance sheet columns found. Using all numeric columns.")
        balance_sheet_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_bs = df[balance_sheet_cols].dropna()
    print(f"Balance sheet columns: {len(balance_sheet_cols)}")

    # Initialize forecaster
    forecaster = BalanceSheetForecaster(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        lstm_units=[128, 64],
        dense_units=[64, 32]
    )

    # Preprocess data
    normalized_data = forecaster.preprocess_data(df_bs)

    # Create sequences
    X, y = create_sequences(normalized_data,
                           sequence_length=sequence_length,
                           forecast_horizon=forecast_horizon)

    print(f"Created sequences: X={X.shape}, y={y.shape}")

    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Build and train model
    forecaster.build_model(input_dim=X.shape[-1])
    print("\nTraining model...")

    history = forecaster.train(X_train, y_train, X_test, y_test,
                              epochs=100, batch_size=32)

    # Evaluate
    print("\nEvaluating model...")
    evaluator = ModelEvaluator()

    y_pred = forecaster.model.predict(X_test)
    metrics = evaluator.calculate_metrics(y_test, y_pred)

    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    # Denormalize predictions for identity validation
    y_pred_denorm = forecaster.scaler.inverse_transform(y_pred)
    y_test_denorm = forecaster.scaler.inverse_transform(y_test)

    pred_df = pd.DataFrame(y_pred_denorm, columns=balance_sheet_cols)
    identity_validation = evaluator.validate_accounting_identities(pred_df)

    print("\nAccounting Identity Validation:")
    for identity, results in identity_validation.items():
        print(f"  {identity}:")
        for key, value in results.items():
            if not isinstance(value, list):
                print(f"    {key}: {value}")

    # Temporal consistency
    temporal_metrics = evaluator.evaluate_temporal_consistency(y_pred_denorm)
    print("\nTemporal Consistency:")
    for metric, value in temporal_metrics.items():
        if not isinstance(value, (list, np.ndarray)):
            print(f"  {metric}: {value:.4f}")

    return {
        'ticker': ticker,
        'metrics': metrics,
        'identity_validation': identity_validation,
        'temporal_metrics': temporal_metrics,
        'training_history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    }


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train balance sheet forecasting model')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--data-path', default='data', help='Path to data directory')
    parser.add_argument('--sequence-length', type=int, default=4,
                       help='Sequence length for LSTM')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Train model
    results = train_model(
        ticker=args.ticker,
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        test_size=args.test_size
    )

    # Save results
    output_file = output_dir / f"{args.ticker}_training_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
