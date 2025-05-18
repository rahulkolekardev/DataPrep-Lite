import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_numeric_columns

class MinMaxScalerWrapper(BaseTransformer):
    """Wrapper for scikit-learn's MinMaxScaler."""
    def __init__(self, columns_to_process: Optional[List[str]] = None, feature_range=(0, 1)):
        super().__init__(columns_to_process)
        self.feature_range = feature_range
        self.scaler_: Optional[MinMaxScaler] = None

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            selected_cols = identify_numeric_columns(X)
            if not selected_cols:
                raise ValueError("MinMaxScalerWrapper: No numeric columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            non_numeric = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"MinMaxScalerWrapper: Columns {non_numeric} are not numeric.")
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MinMaxScalerWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        if not self._processed_columns:
            self.scaler_ = None
            return self

        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
        # MinMaxScaler handles NaNs by ignoring them in fit and keeping them in transform.
        self.scaler_.fit(X[self._processed_columns].values) # Pass NumPy array
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        if not self.scaler_ or not self._processed_columns:
            return X_transformed
        
        # Ensure columns exist
        current_cols = [c for c in self._processed_columns if c in X_transformed.columns]
        if not current_cols: return X_transformed # Nothing to scale

        scaled_data = self.scaler_.transform(X_transformed[current_cols].values)
        X_transformed[current_cols] = scaled_data
        return X_transformed
    # get_feature_names_out from BaseTransformer is suitable.

class StandardScalerWrapper(BaseTransformer):
    """Wrapper for scikit-learn's StandardScaler."""
    def __init__(self, columns_to_process: Optional[List[str]] = None, with_mean=True, with_std=True):
        super().__init__(columns_to_process)
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler_: Optional[StandardScaler] = None

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        # Similar to MinMaxScalerWrapper's implementation
        if self.columns_to_process is None:
            selected_cols = identify_numeric_columns(X)
            if not selected_cols:
                raise ValueError("StandardScalerWrapper: No numeric columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            non_numeric = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"StandardScalerWrapper: Columns {non_numeric} are not numeric.")
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StandardScalerWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        if not self._processed_columns:
            self.scaler_ = None
            return self

        self.scaler_ = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.scaler_.fit(X[self._processed_columns].values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        if not self.scaler_ or not self._processed_columns:
            return X_transformed

        current_cols = [c for c in self._processed_columns if c in X_transformed.columns]
        if not current_cols: return X_transformed

        scaled_data = self.scaler_.transform(X_transformed[current_cols].values)
        X_transformed[current_cols] = scaled_data
        return X_transformed
    # get_feature_names_out from BaseTransformer is suitable.