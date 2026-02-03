"""
Sklearn Data Preprocessing Pipeline
====================================

Standardized data preprocessing following Section III.A recommendations.

Implements unified sklearn preprocessing pipelines for:
- Numerical feature scaling and imputation
- Categorical feature encoding
- Feature engineering
- Data splitting and validation

Section III.A: "Scikit-learn is the standard library for traditional
machine learning algorithms and data preprocessing (e.g., scaling,
missing value imputation), laying the foundation for workflows by
providing a unified API."
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class FinancialDataPreprocessor:
    """
    Standardized sklearn preprocessing for financial tabular data.

    Implements Section III.A unified preprocessing API.
    """

    def __init__(self,
                 numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 scaler_type: str = 'robust'):
        """
        Initialize financial data preprocessor.

        Args:
            numeric_features: List of numerical column names
            categorical_features: List of categorical column names
            scaler_type: 'standard', 'robust', or 'minmax'
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.feature_names_out = None

    def create_pipeline(self, for_gbdt: bool = True) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.

        Args:
            for_gbdt: If True, use ordinal encoding for GBDT models.
                     If False, use one-hot encoding for linear models.

        Returns:
            Sklearn ColumnTransformer pipeline
        """
        # Numerical pipeline
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            scaler = RobustScaler()  # Robust to outliers (recommended for financial data)
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
            ('scaler', scaler)
        ])

        # Categorical pipeline
        # For GBDT: Use ordinal encoding (preserves order, efficient)
        # For linear models: Use one-hot encoding
        if for_gbdt:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'  # Keep other columns as-is
        )

        self.preprocessor = preprocessor
        return preprocessor

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data.

        Args:
            X: Input DataFrame

        Returns:
            Transformed numpy array
        """
        if self.preprocessor is None:
            self.create_pipeline()

        X_transformed = self.preprocessor.fit_transform(X)

        # Store feature names for interpretability
        self._extract_feature_names()

        return X_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Input DataFrame

        Returns:
            Transformed numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        return self.preprocessor.transform(X)

    def _extract_feature_names(self):
        """Extract feature names after transformation."""
        try:
            self.feature_names_out = self.preprocessor.get_feature_names_out()
        except:
            # Fallback if get_feature_names_out not available
            self.feature_names_out = (
                self.numeric_features +
                self.categorical_features
            )

    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation.

        Returns:
            List of feature names
        """
        if self.feature_names_out is None:
            self._extract_feature_names()
        return list(self.feature_names_out)

    def save_pipeline(self, filepath: str):
        """Save preprocessing pipeline to file."""
        joblib.dump({
            'preprocessor': self.preprocessor,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'scaler_type': self.scaler_type,
            'feature_names_out': self.feature_names_out
        }, filepath)
        print(f"✓ Preprocessing pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load preprocessing pipeline from file."""
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.scaler_type = data['scaler_type']
        self.feature_names_out = data.get('feature_names_out')
        print(f"✓ Preprocessing pipeline loaded from {filepath}")


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for financial feature engineering.

    Can be integrated into sklearn pipelines.
    """

    def __init__(self, create_ratios: bool = True):
        """
        Initialize feature engineer.

        Args:
            create_ratios: Whether to create financial ratios
        """
        self.create_ratios = create_ratios

    def fit(self, X, y=None):
        """Fit transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with financial feature engineering.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame with new features
        """
        X_new = X.copy()

        if self.create_ratios:
            # Example financial ratios
            if 'total_debt' in X.columns and 'total_equity' in X.columns:
                X_new['debt_to_equity'] = X['total_debt'] / (X['total_equity'] + 1e-8)

            if 'current_assets' in X.columns and 'current_liabilities' in X.columns:
                X_new['current_ratio'] = X['current_assets'] / (X['current_liabilities'] + 1e-8)

            if 'net_income' in X.columns and 'total_assets' in X.columns:
                X_new['roa'] = X['net_income'] / (X['total_assets'] + 1e-8)

        return X_new


def create_train_test_split(X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.Series, np.ndarray],
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            random_state: int = 42) -> Tuple:
    """
    Create train/validation/test splits.

    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        val_size: Validation set proportion (from training set)
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Demonstrate sklearn preprocessing pipeline."""
    print("Financial Data Preprocessing - Section III.A Implementation")
    print("=" * 80)

    # Create sample financial data
    print("\n1. Creating sample financial dataset...")
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        # Numerical features
        'total_assets': np.random.uniform(100000, 1000000, n_samples),
        'total_debt': np.random.uniform(50000, 500000, n_samples),
        'revenue': np.random.uniform(200000, 2000000, n_samples),
        'net_income': np.random.uniform(-50000, 200000, n_samples),

        # Categorical features
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),

        # Target
        'credit_approved': np.random.choice([0, 1], n_samples)
    })

    # Add some missing values
    df.loc[df.sample(50).index, 'net_income'] = np.nan
    df.loc[df.sample(30).index, 'industry'] = np.nan

    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")

    # Define feature columns
    numeric_features = ['total_assets', 'total_debt', 'revenue', 'net_income']
    categorical_features = ['industry', 'region']

    X = df[numeric_features + categorical_features]
    y = df['credit_approved']

    # Create preprocessor
    print("\n2. Creating sklearn preprocessing pipeline...")
    preprocessor = FinancialDataPreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler_type='robust'  # Robust to outliers
    )

    # Fit and transform
    print("\n3. Fitting and transforming data...")
    X_transformed = preprocessor.fit_transform(X)

    print(f"   Transformed shape: {X_transformed.shape}")
    print(f"   Feature names: {preprocessor.get_feature_names()}")

    # Create splits
    print("\n4. Creating train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
        X_transformed, y, test_size=0.2, val_size=0.1
    )

    print(f"   Train set: {X_train.shape}")
    print(f"   Val set:   {X_val.shape}")
    print(f"   Test set:  {X_test.shape}")

    # Save pipeline
    print("\n5. Saving preprocessing pipeline...")
    preprocessor.save_pipeline('models/preprocessing_pipeline.pkl')

    print("\n✓ Preprocessing pipeline demonstration complete!")
    print("\nKEY FEATURES (Section III.A):")
    print("  - Unified sklearn API for all preprocessing")
    print("  - Robust scaling (handles financial outliers)")
    print("  - Missing value imputation")
    print("  - Ordinal encoding for GBDT models")
    print("  - Pipeline can be saved/loaded for production")


if __name__ == "__main__":
    main()
