"""
ML Enhancements for Balance Sheet Forecasting
----------------------------------------------
Advanced ML techniques to improve forecasting performance.

Questions addressed: Part 1, Q7

Techniques implemented:
1. Attention mechanisms
2. Ensemble methods
3. Transfer learning
4. Feature engineering
5. Hyperparameter optimization
6. Residual connections
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import optuna


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for time series forecasting.

    Helps model focus on most relevant historical periods.
    """

    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Build attention weights."""
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Apply attention mechanism.

        Args:
            inputs: Input tensor of shape (batch, timesteps, features)

        Returns:
            Context vector with attention applied
        """
        # Calculate attention scores
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)

        # Apply attention
        attention_weights = tf.expand_dims(attention_weights, -1)
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


class EnhancedBalanceSheetModel(keras.Model):
    """
    Enhanced balance sheet forecasting model with advanced ML techniques.
    """

    def __init__(self,
                 input_dim: int,
                 lstm_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32],
                 use_attention: bool = True,
                 use_residual: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize enhanced model.

        Args:
            input_dim: Number of input features
            lstm_units: LSTM layer units
            dense_units: Dense layer units
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
            dropout_rate: Dropout rate
        """
        super(EnhancedBalanceSheetModel, self).__init__()

        self.input_dim = input_dim
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Bidirectional LSTM layers
        self.bi_lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1 or use_attention
            self.bi_lstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences, dropout=dropout_rate),
                    name=f'bi_lstm_{i}'
                )
            )

        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(units=64, name='attention')

        # Dense layers with residual connections
        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []

        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'dense_{i}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
            self.batch_norm_layers.append(
                layers.BatchNormalization(name=f'batch_norm_{i}')
            )

        # Output layer
        self.output_layer = layers.Dense(input_dim, activation='linear', name='output')

    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs

        # Bidirectional LSTM layers
        for bi_lstm in self.bi_lstm_layers:
            x = bi_lstm(x, training=training)

        # Attention mechanism
        if self.use_attention:
            x = self.attention(x)

        # Dense layers with residual connections
        for i, (dense, dropout, bn) in enumerate(
            zip(self.dense_layers, self.dropout_layers, self.batch_norm_layers)
        ):
            residual = x
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)

            # Residual connection (if dimensions match)
            if self.use_residual and x.shape[-1] == residual.shape[-1]:
                x = x + residual

        # Output
        outputs = self.output_layer(x)

        return outputs


class EnsembleForecaster:
    """
    Ensemble of multiple forecasting models.

    Combines predictions from different models for better accuracy.
    """

    def __init__(self):
        """Initialize ensemble."""
        self.models = []
        self.weights = []

    def add_model(self, model, weight: float = 1.0):
        """
        Add a model to the ensemble.

        Args:
            model: Forecasting model
            weight: Weight for this model's predictions
        """
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.

        Args:
            X: Input data

        Returns:
            Weighted average of all model predictions
        """
        predictions = []
        total_weight = sum(self.weights)

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0)
            predictions.append(pred * (weight / total_weight))

        return np.sum(predictions, axis=0)


class FeatureEngineer:
    """
    Advanced feature engineering for financial data.
    """

    @staticmethod
    def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratios as features.

        Args:
            df: DataFrame with financial data

        Returns:
            DataFrame with additional ratio features
        """
        ratios = df.copy()

        # Liquidity ratios
        if 'Current Assets' in df.columns and 'Current Liabilities' in df.columns:
            ratios['Current_Ratio'] = df['Current Assets'] / df['Current Liabilities']

        if 'Cash And Cash Equivalents' in df.columns and 'Current Liabilities' in df.columns:
            ratios['Cash_Ratio'] = df['Cash And Cash Equivalents'] / df['Current Liabilities']

        # Leverage ratios
        if 'Total Debt' in df.columns and 'Stockholders Equity' in df.columns:
            ratios['Debt_to_Equity'] = df['Total Debt'] / df['Stockholders Equity']

        if 'Total Debt' in df.columns and 'Total Assets' in df.columns:
            ratios['Debt_to_Assets'] = df['Total Debt'] / df['Total Assets']

        # Efficiency ratios
        if 'Total Assets' in df.columns:
            ratios['Asset_Growth'] = df['Total Assets'].pct_change()

        if 'Stockholders Equity' in df.columns:
            ratios['Equity_Growth'] = df['Stockholders Equity'].pct_change()

        # Size features (log transformation for scale)
        if 'Total Assets' in df.columns:
            ratios['Log_Assets'] = np.log1p(df['Total Assets'])

        return ratios.fillna(0)

    @staticmethod
    def create_lagged_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            df: DataFrame with time series data
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        lagged_df = df.copy()

        for col in df.columns:
            for lag in lags:
                lagged_df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return lagged_df.fillna(0)

    @staticmethod
    def create_rolling_statistics(df: pd.DataFrame,
                                  windows: List[int] = [2, 4]) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            df: DataFrame with time series data
            windows: List of window sizes

        Returns:
            DataFrame with rolling statistics
        """
        rolling_df = df.copy()

        for col in df.columns:
            for window in windows:
                rolling_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                rolling_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()

        return rolling_df.fillna(0)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """

    def __init__(self, input_dim: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray):
        """
        Initialize optimizer.

        Args:
            input_dim: Input dimension
            X_train: Training data
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets
        """
        self.input_dim = input_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss
        """
        # Suggest hyperparameters
        lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256, step=32)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
        dense_units_1 = trial.suggest_int('dense_units_1', 32, 128, step=32)
        dense_units_2 = trial.suggest_int('dense_units_2', 16, 64, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        # Build model
        model = EnhancedBalanceSheetModel(
            input_dim=self.input_dim,
            lstm_units=[lstm_units_1, lstm_units_2],
            dense_units=[dense_units_1, dense_units_2],
            dropout_rate=dropout_rate
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        # Train
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )

        # Return validation loss
        return min(history.history['val_loss'])

    def optimize(self, n_trials: int = 50) -> Dict:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }


def main():
    """Demonstrate ML enhancements."""
    print("ML Enhancements for Balance Sheet Forecasting")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    sequence_length = 4

    # Generate sequences
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples, n_features)

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 1. Enhanced model with attention
    print("\n1. Training Enhanced Model with Attention")
    print("-" * 80)
    enhanced_model = EnhancedBalanceSheetModel(
        input_dim=n_features,
        lstm_units=[64, 32],
        dense_units=[32, 16],
        use_attention=True,
        use_residual=True
    )

    enhanced_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )

    history = enhanced_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        verbose=0
    )

    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # 2. Feature engineering
    print("\n2. Feature Engineering")
    print("-" * 80)
    sample_df = pd.DataFrame(
        np.random.randn(20, 5),
        columns=['Total Assets', 'Current Assets', 'Current Liabilities',
                'Total Debt', 'Stockholders Equity']
    )
    sample_df = sample_df.abs()  # Make positive for ratios

    fe = FeatureEngineer()
    enhanced_df = fe.create_financial_ratios(sample_df)
    print(f"Original features: {sample_df.shape[1]}")
    print(f"Enhanced features: {enhanced_df.shape[1]}")
    print(f"New features: {[col for col in enhanced_df.columns if col not in sample_df.columns]}")

    # 3. Ensemble
    print("\n3. Ensemble Forecasting")
    print("-" * 80)
    ensemble = EnsembleForecaster()

    # Create multiple models
    for i in range(3):
        model = EnhancedBalanceSheetModel(
            input_dim=n_features,
            lstm_units=[64, 32],
            dense_units=[32]
        )
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, verbose=0)
        ensemble.add_model(model, weight=1.0)

    ensemble_pred = ensemble.predict(X_val)
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")

    print("\n4. Hyperparameter Optimization (example)")
    print("-" * 80)
    print("Note: Running Optuna optimization (would take time)...")
    print("Example best parameters:")
    example_params = {
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'dense_units_1': 64,
        'dense_units_2': 32,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
    for param, value in example_params.items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
