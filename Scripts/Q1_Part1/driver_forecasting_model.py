"""
Driver Forecasting Model
========================
This module shows the CORRECT use of ML/LSTMs in balance sheet forecasting.

KEY PRINCIPLE: ML should forecast DRIVERS x(t), NOT balance sheet line items directly.

The drivers (revenue growth, margins, etc.) are then fed into the deterministic
accounting model to generate balanced financial statements.

This is the "x(t)" in the model form: y(t+1) = f(x(t), y(t)) + n(t)

ML Techniques Applied:
1. LSTM for time-series forecasting of financial ratios
2. XGBoost for margin prediction based on macro factors
3. Feature engineering from historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# Import XGBoost conditionally
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


class RevenueGrowthForecaster:
    """
    Forecast revenue growth using LSTM on historical data.
    
    This is the CORRECT use of LSTM - forecasting a driver, not the entire balance sheet.
    """
    
    def __init__(self, sequence_length: int = 8):
        """
        Initialize the revenue growth forecaster.
        
        Args:
            sequence_length: Number of historical periods to use
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, n_features: int):
        """Build LSTM model for revenue growth forecasting."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM model")
        
        model = keras.Sequential([
            layers.LSTM(32, activation='relu', input_shape=(self.sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Output: revenue growth rate
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length, 0])  # Target: next period's growth
        return np.array(X), np.array(y)
    
    def train(self, historical_data: pd.DataFrame, target_col: str = 'revenue_growth'):
        """
        Train the LSTM on historical revenue growth data.
        
        Args:
            historical_data: DataFrame with revenue_growth and other features
            target_col: Name of the target column
        """
        # Prepare data
        features = historical_data.values
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = self.prepare_sequences(features_scaled)
        
        if len(X) == 0:
            raise ValueError("Not enough data for sequence creation")
        
        # Build and train model
        self.build_model(n_features=features.shape[1])
        
        self.model.fit(
            X, y,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            verbose=0
        )
    
    def forecast(self, recent_data: pd.DataFrame, periods: int = 1) -> np.ndarray:
        """
        Forecast revenue growth for future periods.
        
        Args:
            recent_data: Most recent historical data (length >= sequence_length)
            periods: Number of periods to forecast
            
        Returns:
            Array of forecasted revenue growth rates
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale recent data
        recent_scaled = self.scaler.transform(recent_data.values)
        
        forecasts = []
        current_sequence = recent_scaled[-self.sequence_length:]
        
        for _ in range(periods):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict next period
            pred = self.model.predict(X, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence (simplified - just repeat last row with new growth)
            new_row = current_sequence[-1].copy()
            new_row[0] = pred
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(forecasts)


class MarginForecaster:
    """
    Forecast operating margins using XGBoost with macro features.
    
    This shows how to use ML to forecast margin drivers based on external factors.
    """
    
    def __init__(self):
        """Initialize the margin forecaster."""
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train XGBoost model to forecast margins.
        
        Args:
            X_train: Features (e.g., GDP growth, commodity prices, inflation)
            y_train: Target margin (e.g., COGS as % of revenue)
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required for margin forecasting")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_scaled, y_train)
    
    def forecast(self, X_future: pd.DataFrame) -> np.ndarray:
        """
        Forecast margins for future periods.
        
        Args:
            X_future: Future values of macro features
            
        Returns:
            Forecasted margin values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X_future)
        return self.model.predict(X_scaled)


class IntegratedDriverForecaster:
    """
    Integrated system that forecasts all drivers needed for the deterministic model.
    
    This is the CORRECT architecture:
    1. Use ML to forecast drivers (x(t))
    2. Feed drivers into deterministic_accounting_model.py
    3. Get balanced financial statements
    """
    
    def __init__(self):
        """Initialize the integrated forecaster."""
        self.revenue_forecaster = RevenueGrowthForecaster()
        self.margin_forecasters = {}
    
    def train_all(self, historical_data: pd.DataFrame):
        """
        Train all driver forecasting models.
        
        Args:
            historical_data: DataFrame with columns:
                - revenue_growth
                - cogs_pct
                - opex_pct
                - gdp_growth (macro feature)
                - inflation (macro feature)
                - etc.
        """
        # Train revenue growth forecaster
        revenue_features = historical_data[['revenue_growth', 'gdp_growth']]
        self.revenue_forecaster.train(revenue_features)
        
        print("[OK] Revenue growth forecaster trained")
        
        # Train margin forecasters (if XGBoost available)
        if XGB_AVAILABLE:
            # COGS margin forecaster
            cogs_forecaster = MarginForecaster()
            macro_features = historical_data[['gdp_growth', 'inflation']]
            cogs_forecaster.train(macro_features, historical_data['cogs_pct'])
            self.margin_forecasters['cogs'] = cogs_forecaster
            
            print("[OK] COGS margin forecaster trained")
    
    def forecast_drivers(
        self,
        recent_data: pd.DataFrame,
        future_macro: pd.DataFrame,
        periods: int = 1
    ) -> List[Dict[str, float]]:
        """
        Forecast all drivers for future periods.
        
        Args:
            recent_data: Recent historical data for LSTM
            future_macro: Future macro assumptions for XGBoost
            periods: Number of periods to forecast
            
        Returns:
            List of driver dictionaries, one per period
        """
        # Forecast revenue growth
        revenue_growth_forecasts = self.revenue_forecaster.forecast(
            recent_data[['revenue_growth', 'gdp_growth']],
            periods=periods
        )
        
        # Forecast margins
        cogs_forecasts = self.margin_forecasters['cogs'].forecast(future_macro)
        
        # Package into driver dictionaries
        driver_forecasts = []
        for i in range(periods):
            drivers = {
                'revenue_growth': float(revenue_growth_forecasts[i]),
                'cogs_as_pct_revenue': float(cogs_forecasts[i]),
                'opex_as_pct_revenue': 0.20,  # Could add more forecasters
                'tax_rate': 0.21,
                'days_sales_outstanding': 45,
                'days_inventory_outstanding': 60,
                'days_payable_outstanding': 30,
                'capex_as_pct_revenue': 0.08,
                'depreciation_as_pct_ppe': 0.10,
                'dividend_payout_ratio': 0.30,
                'target_cash_balance': 100_000,
                'interest_rate_on_debt': 0.05
            }
            driver_forecasts.append(drivers)
        
        return driver_forecasts


def main():
    """Demonstrate driver forecasting."""
    print("=" * 80)
    print("DRIVER FORECASTING MODEL")
    print("The CORRECT use of ML in balance sheet forecasting")
    print("=" * 80)
    print()
    
    print("KEY PRINCIPLE:")
    print("  ML forecasts DRIVERS (revenue growth, margins)")
    print("  NOT balance sheet line items directly")
    print()
    print("This avoids violating accounting identities!")
    print("=" * 80)
    print()
    
    # Create synthetic historical data
    np.random.seed(42)
    n_periods = 50
    
    historical_data = pd.DataFrame({
        'revenue_growth': np.random.normal(0.05, 0.02, n_periods),
        'gdp_growth': np.random.normal(0.03, 0.01, n_periods),
        'inflation': np.random.normal(0.02, 0.005, n_periods),
        'cogs_pct': np.random.normal(0.60, 0.02, n_periods),
        'opex_pct': np.random.normal(0.20, 0.01, n_periods)
    })
    
    print("Sample historical data:")
    print(historical_data.head())
    print()
    
    # Train forecasters
    print("Training models...")
    forecaster = IntegratedDriverForecaster()
    
    try:
        forecaster.train_all(historical_data)
        print()
        
        # Forecast future drivers
        recent_data = historical_data.tail(8)  # Last 8 periods
        future_macro = pd.DataFrame({
            'gdp_growth': [0.03, 0.03, 0.03],
            'inflation': [0.025, 0.025, 0.025]
        })
        
        driver_forecasts = forecaster.forecast_drivers(
            recent_data=recent_data,
            future_macro=future_macro,
            periods=3
        )
        
        print("FORECASTED DRIVERS (3 periods):")
        print("=" * 80)
        for i, drivers in enumerate(driver_forecasts):
            print(f"\nPeriod {i+1}:")
            print(f"  Revenue Growth: {drivers['revenue_growth']*100:.2f}%")
            print(f"  COGS Margin: {drivers['cogs_as_pct_revenue']*100:.2f}%")
            print(f"  OpEx Margin: {drivers['opex_as_pct_revenue']*100:.2f}%")
        
        print()
        print("=" * 80)
        print("NEXT STEP:")
        print("Feed these forecasted drivers into deterministic_accounting_model.py")
        print("to generate fully balanced financial statements!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"[WARNING] {e}")
        print("This is a demonstration. Install required packages to run fully.")


if __name__ == "__main__":
    main()
