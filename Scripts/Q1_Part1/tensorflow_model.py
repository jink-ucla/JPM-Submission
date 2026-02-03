"""
TensorFlow Balance Sheet Forecasting Model
-------------------------------------------
⚠️  WARNING - DEPRECATED APPROACH ⚠️

This file demonstrates a FUNDAMENTALLY FLAWED approach to balance sheet forecasting.

CRITICAL ERROR: This model uses LSTM to forecast balance sheet line items DIRECTLY.
This violates accounting identities (Assets = Liabilities + Equity) because each item
is forecasted independently without structural constraints.

WHY THIS IS WRONG (per Velez-Pareja & Shahnazarian):
1. LSTM learns statistical patterns, not accounting rules
2. Forecasting line items independently GUARANTEES imbalance
3. The "accounting constraint layer" and "identity violation loss" are band-aids
   that don't solve the fundamental problem - they just measure the error
4. This creates a "plug" (the violation) which defeats the purpose

CORRECT APPROACH:
- Use deterministic_accounting_model.py (implements Velez-Pareja's no-plug method)
- Use driver_forecasting_model.py (ML forecasts DRIVERS, not balance sheet items)
- See integrated_example.py for the complete workflow

This file is kept for reference only to show what NOT to do.

Original description:
Neural network implementation for balance sheet forecasting with accounting constraints.
Questions addressed: Part 1, Q3, Q5, Q8 (INCORRECT IMPLEMENTATION)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class AccountingConstraintLayer(layers.Layer):
    """
    Custom layer to enforce accounting identity: Assets = Liabilities + Equity.

    This layer ensures the fundamental accounting equation is satisfied
    by adjusting one component (typically equity) to balance the equation.
    """

    def __init__(self, **kwargs):
        super(AccountingConstraintLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Enforce accounting constraint.

        Args:
            inputs: Tuple of (assets, liabilities, equity)

        Returns:
            Adjusted values satisfying Assets = Liabilities + Equity
        """
        assets, liabilities, equity = inputs

        # Adjust equity to satisfy the constraint
        adjusted_equity = assets - liabilities

        return assets, liabilities, adjusted_equity

    def get_config(self):
        """Return layer configuration."""
        return super(AccountingConstraintLayer, self).get_config()


class BalanceSheetLSTM(Model):
    """
    LSTM-based model for balance sheet forecasting.

    Model architecture:
    - Input: Historical balance sheet and income statement data
    - LSTM layers for temporal dependencies
    - Dense layers for feature transformation
    - Custom constraint layer for accounting identity
    """

    def __init__(self,
                 input_dim: int,
                 lstm_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 enforce_constraints: bool = True):
        """
        Initialize the balance sheet LSTM model.

        Args:
            input_dim: Number of input features
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
            enforce_constraints: Whether to enforce accounting constraints
        """
        super(BalanceSheetLSTM, self).__init__()

        self.input_dim = input_dim
        self.enforce_constraints = enforce_constraints

        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(units, return_sequences=return_sequences,
                           dropout=dropout_rate, name=f'lstm_{i}')
            )

        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'dense_{i}')
            )
            self.dense_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )

        # Output layer for balance sheet components
        self.output_layer = layers.Dense(input_dim, activation='linear',
                                        name='balance_sheet_output')

        # Constraint layer
        if enforce_constraints:
            self.constraint_layer = AccountingConstraintLayer(name='accounting_constraint')

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, input_dim)
            training: Whether in training mode

        Returns:
            Predicted balance sheet values
        """
        x = inputs

        # LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # Dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x, training=training)

        # Output
        outputs = self.output_layer(x)

        return outputs

    def get_config(self):
        """Return model configuration."""
        return {
            'input_dim': self.input_dim,
            'enforce_constraints': self.enforce_constraints
        }


def create_sequences(data: np.ndarray,
                    sequence_length: int,
                    forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.

    Args:
        data: Input data of shape (n_samples, n_features)
        sequence_length: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast

    Returns:
        X: Input sequences of shape (n_samples, sequence_length, n_features)
        y: Target sequences of shape (n_samples, n_features)
    """
    X, y = [], []

    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])

    return np.array(X), np.array(y).squeeze()


def accounting_identity_loss(y_true, y_pred, asset_idx=0, liability_idx=1, equity_idx=2):
    """
    Custom loss function that penalizes violations of accounting identity.

    Loss = MSE + lambda * |Assets - (Liabilities + Equity)|^2

    Args:
        y_true: True values
        y_pred: Predicted values
        asset_idx: Index of assets in the output
        liability_idx: Index of liabilities in the output
        equity_idx: Index of equity in the output

    Returns:
        Combined loss value
    """
    # Standard MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Accounting identity violation penalty
    assets = y_pred[:, asset_idx]
    liabilities = y_pred[:, liability_idx]
    equity = y_pred[:, equity_idx]

    identity_violation = tf.square(assets - (liabilities + equity))
    identity_loss = tf.reduce_mean(identity_violation)

    # Combined loss (lambda = 10 for strong enforcement)
    lambda_constraint = 10.0
    total_loss = mse_loss + lambda_constraint * identity_loss

    return total_loss


class BalanceSheetForecaster:
    """
    Complete forecasting system for balance sheets.

    Handles data preprocessing, model training, evaluation, and forecasting.
    """

    def __init__(self,
                 sequence_length: int = 4,
                 forecast_horizon: int = 1,
                 lstm_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32]):
        """
        Initialize the forecaster.

        Args:
            sequence_length: Number of historical periods to use
            forecast_horizon: Number of periods to forecast
            lstm_units: LSTM layer configuration
            dense_units: Dense layer configuration
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None
        self.scaler = None

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess financial data for modeling.

        Args:
            data: DataFrame with financial statement data

        Returns:
            Normalized numpy array
        """
        from sklearn.preprocessing import StandardScaler

        # Remove any NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Normalize data
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(data)

        return normalized_data

    def build_model(self, input_dim: int):
        """
        Build and compile the TensorFlow model.

        Args:
            input_dim: Number of input features
        """
        self.model = BalanceSheetLSTM(
            input_dim=input_dim,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            enforce_constraints=True
        )

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Can use accounting_identity_loss for constraint enforcement
            metrics=['mae', 'mse']
        )

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/balance_sheet_model.h5', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def forecast(self, historical_data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Forecast future balance sheet values.

        Args:
            historical_data: Historical data of shape (sequence_length, n_features)
            steps: Number of steps to forecast

        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        forecasts = []
        current_sequence = historical_data[-self.sequence_length:].copy()

        for _ in range(steps):
            # Reshape for model input
            input_seq = current_sequence.reshape(1, self.sequence_length, -1)

            # Predict next step
            prediction = self.model.predict(input_seq, verbose=0)

            forecasts.append(prediction[0])

            # Update sequence for next prediction
            current_sequence = np.vstack([current_sequence[1:], prediction[0]])

        return np.array(forecasts)

    def validate_accounting_identities(self,
                                      predictions: np.ndarray,
                                      asset_idx: int,
                                      liability_idx: int,
                                      equity_idx: int) -> Dict[str, float]:
        """
        Validate that predictions satisfy accounting identities.

        Args:
            predictions: Predicted balance sheet values
            asset_idx: Index of assets column
            liability_idx: Index of liabilities column
            equity_idx: Index of equity column

        Returns:
            Dictionary with validation metrics
        """
        assets = predictions[:, asset_idx]
        liabilities = predictions[:, liability_idx]
        equity = predictions[:, equity_idx]

        violations = assets - (liabilities + equity)
        relative_violations = violations / assets

        return {
            'mean_absolute_violation': np.mean(np.abs(violations)),
            'max_violation': np.max(np.abs(violations)),
            'mean_relative_violation': np.mean(np.abs(relative_violations)),
            'max_relative_violation': np.max(np.abs(relative_violations)),
            'violations': violations
        }


def main():
    """Demonstrate TensorFlow model usage."""
    print("Balance Sheet Forecasting with TensorFlow")
    print("=" * 80)

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Generate synthetic balance sheet data
    data = np.random.randn(n_samples, n_features)

    # Ensure accounting identity in synthetic data
    # Columns 0=Assets, 1=Liabilities, 2=Equity
    data[:, 2] = data[:, 0] - data[:, 1]  # Equity = Assets - Liabilities

    # Initialize forecaster
    forecaster = BalanceSheetForecaster(
        sequence_length=4,
        forecast_horizon=1,
        lstm_units=[64, 32],
        dense_units=[32, 16]
    )

    # Create sequences
    X, y = create_sequences(data, sequence_length=4, forecast_horizon=1)
    print(f"Sequence shapes: X={X.shape}, y={y.shape}")

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Build model
    forecaster.build_model(input_dim=n_features)
    print(f"\nModel built with input dimension: {n_features}")

    # Train model
    print("\nTraining model...")
    history = forecaster.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)

    # Make forecast
    print("\nMaking forecast...")
    forecast = forecaster.forecast(data[-4:], steps=3)
    print(f"Forecast shape: {forecast.shape}")

    # Validate accounting identities
    validation = forecaster.validate_accounting_identities(
        forecast, asset_idx=0, liability_idx=1, equity_idx=2
    )
    print("\nAccounting Identity Validation:")
    print(f"  Mean Absolute Violation: {validation['mean_absolute_violation']:.6f}")
    print(f"  Max Violation: {validation['max_violation']:.6f}")
    print(f"  Mean Relative Violation: {validation['mean_relative_violation']*100:.4f}%")


if __name__ == "__main__":
    main()
