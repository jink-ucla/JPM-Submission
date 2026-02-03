"""
Utility Functions for Balance Sheet Forecasting
------------------------------------------------
Helper functions for data processing, validation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def validate_balance_sheet_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate balance sheet data quality.

    Args:
        df: Balance sheet DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.to_dict(),
        'negative_values': {},
        'zero_values': {},
    }

    # Check for negative values in key fields
    positive_fields = ['Total Assets', 'Current Assets', 'Total Liabilities']
    for field in positive_fields:
        if field in df.columns:
            negative_count = (df[field] < 0).sum()
            if negative_count > 0:
                validation['negative_values'][field] = negative_count

    # Check for zero values
    for col in df.select_dtypes(include=[np.number]).columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            validation['zero_values'][col] = zero_count

    return validation


def check_accounting_identities(df: pd.DataFrame,
                                tolerance: float = 1e-2) -> pd.DataFrame:
    """
    Check if accounting identities hold.

    Args:
        df: Balance sheet DataFrame
        tolerance: Tolerance for relative error

    Returns:
        DataFrame with check results
    """
    results = []

    for idx in df.index:
        row = df.loc[idx]
        checks = {'date': idx}

        # 1. Assets = Liabilities + Equity
        if all(k in row for k in ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']):
            assets = row['Total Assets']
            liabilities = row['Total Liabilities Net Minority Interest']
            equity = row['Stockholders Equity']

            diff = assets - (liabilities + equity)
            rel_error = abs(diff / assets) if assets != 0 else 0

            checks['fundamental_equation_diff'] = diff
            checks['fundamental_equation_rel_error'] = rel_error
            checks['fundamental_equation_valid'] = rel_error < tolerance

        # 2. Current Ratio calculation
        if all(k in row for k in ['Current Assets', 'Current Liabilities']):
            checks['current_ratio'] = row['Current Assets'] / row['Current Liabilities']

        # 3. Quick Ratio
        if all(k in row for k in ['Cash And Cash Equivalents', 'Accounts Receivable', 'Current Liabilities']):
            quick_assets = row['Cash And Cash Equivalents'] + row.get('Accounts Receivable', 0)
            checks['quick_ratio'] = quick_assets / row['Current Liabilities']

        results.append(checks)

    return pd.DataFrame(results)


def plot_forecast_vs_actual(actual: np.ndarray,
                           forecast: np.ndarray,
                           feature_names: List[str],
                           save_path: Optional[str] = None):
    """
    Plot forecast vs actual values for multiple features.

    Args:
        actual: Actual values (n_samples, n_features)
        forecast: Forecasted values (n_samples, n_features)
        feature_names: Names of features
        save_path: Path to save the plot
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
        if i < actual.shape[1]:
            ax.plot(actual[:, i], 'b-', label='Actual', linewidth=2)
            ax.plot(forecast[:, i], 'r--', label='Forecast', linewidth=2, alpha=0.7)
            ax.set_title(feature_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_training_history(history, save_path: Optional[str] = None):
    """
    Plot training and validation loss.

    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE (if available)
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], 'b-', label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot correlation matrix of financial features.

    Args:
        df: DataFrame with financial data
        save_path: Path to save the plot
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation
    corr = numeric_df.corr()

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Financial Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def calculate_financial_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a composite financial health score.

    Args:
        df: DataFrame with financial ratios

    Returns:
        Series of health scores (0-100)
    """
    scores = pd.Series(0.0, index=df.index)
    max_score = 0

    # Current Ratio (weight: 20 points)
    # Healthy: > 1.5, Warning: 1.0-1.5, Unhealthy: < 1.0
    if 'Current Assets' in df.columns and 'Current Liabilities' in df.columns:
        current_ratio = df['Current Assets'] / df['Current Liabilities']
        scores += np.clip((current_ratio - 0.5) * 20, 0, 20)
        max_score += 20

    # Debt to Equity (weight: 20 points)
    # Healthy: < 1.0, Warning: 1.0-2.0, Unhealthy: > 2.0
    if 'Total Debt' in df.columns and 'Stockholders Equity' in df.columns:
        debt_to_equity = df['Total Debt'] / df['Stockholders Equity']
        scores += np.clip((2.0 - debt_to_equity) * 10, 0, 20)
        max_score += 20

    # Asset Growth (weight: 20 points)
    if 'Total Assets' in df.columns:
        asset_growth = df['Total Assets'].pct_change()
        # Positive growth is good, up to 20% = max score
        scores += np.clip(asset_growth * 100, 0, 20)
        max_score += 20

    # Cash Position (weight: 20 points)
    if 'Cash And Cash Equivalents' in df.columns and 'Total Assets' in df.columns:
        cash_ratio = df['Cash And Cash Equivalents'] / df['Total Assets']
        # Healthy cash position: 10-30% of assets
        scores += np.clip(cash_ratio * 100, 0, 20)
        max_score += 20

    # Equity Strength (weight: 20 points)
    if 'Stockholders Equity' in df.columns and 'Total Assets' in df.columns:
        equity_ratio = df['Stockholders Equity'] / df['Total Assets']
        # Strong equity: > 50% of assets
        scores += np.clip(equity_ratio * 40, 0, 20)
        max_score += 20

    # Normalize to 0-100
    if max_score > 0:
        scores = (scores / max_score) * 100

    return scores


def export_results_to_excel(results: Dict[str, pd.DataFrame],
                           filepath: str):
    """
    Export multiple result DataFrames to Excel with multiple sheets.

    Args:
        results: Dictionary mapping sheet names to DataFrames
        filepath: Output Excel file path
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"Results exported to: {filepath}")


def summarize_model_performance(metrics: Dict[str, float]) -> str:
    """
    Create a text summary of model performance.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        Formatted summary string
    """
    summary = "Model Performance Summary\n"
    summary += "=" * 60 + "\n"

    # Group metrics by type
    accuracy_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    identity_metrics = [k for k in metrics.keys() if 'identity' in k.lower()]

    summary += "\nAccuracy Metrics:\n"
    summary += "-" * 60 + "\n"
    for metric in accuracy_metrics:
        if metric in metrics:
            summary += f"  {metric:20s}: {metrics[metric]:12.4f}\n"

    if identity_metrics:
        summary += "\nAccounting Identity Compliance:\n"
        summary += "-" * 60 + "\n"
        for metric in identity_metrics:
            summary += f"  {metric:40s}: {metrics[metric]}\n"

    return summary


def main():
    """Demonstrate utility functions."""
    print("Balance Sheet Forecasting Utilities")
    print("=" * 80)

    # Create sample data
    sample_data = pd.DataFrame({
        'Total Assets': [1000, 1100, 1200, 1300],
        'Total Liabilities Net Minority Interest': [400, 420, 440, 460],
        'Stockholders Equity': [600, 680, 760, 840],
        'Current Assets': [500, 550, 600, 650],
        'Current Liabilities': [200, 210, 220, 230],
        'Cash And Cash Equivalents': [100, 120, 140, 160],
    }, index=pd.date_range('2020', periods=4, freq='Y'))

    # Validate data
    print("\n1. Data Validation")
    print("-" * 80)
    validation = validate_balance_sheet_data(sample_data)
    print(f"Data shape: {validation['shape']}")
    print(f"Missing values: {sum(validation['missing_values'].values())}")

    # Check accounting identities
    print("\n2. Accounting Identity Checks")
    print("-" * 80)
    checks = check_accounting_identities(sample_data)
    print(checks.to_string())

    # Calculate health score
    print("\n3. Financial Health Score")
    print("-" * 80)
    health_scores = calculate_financial_health_score(sample_data)
    print(health_scores.to_string())

    # Performance summary
    print("\n4. Model Performance Summary")
    print("-" * 80)
    sample_metrics = {
        'MSE': 0.0123,
        'RMSE': 0.1109,
        'MAE': 0.0856,
        'MAPE': 2.45,
        'R2': 0.9567
    }
    summary = summarize_model_performance(sample_metrics)
    print(summary)


if __name__ == "__main__":
    main()
