import pandas as pd
import numpy as np
from typing import List, Optional
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_numeric_columns

class OutlierIQRHandler(BaseTransformer):
    """Handles outliers in numerical columns using IQR. Can cap or remove rows."""
    def __init__(self,
                 columns_to_process: Optional[List[str]] = None,
                 factor: float = 1.5,
                 action: str = 'cap'): # 'cap' or 'remove_rows'
        super().__init__(columns_to_process)
        if factor <= 0:
            raise ValueError("Factor must be positive.")
        if action not in ['cap', 'remove_rows']:
            raise ValueError("Action must be 'cap' or 'remove_rows'.")
        self.factor = factor
        self.action = action
        self.bounds_: dict = {} # Stores (lower_bound, upper_bound) for each processed column

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            selected_cols = identify_numeric_columns(X)
            if not selected_cols:
                raise ValueError("OutlierIQRHandler: No numeric columns found for processing.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X) # Basic existence check
            # Further validation for numeric type
            non_numeric = [col for col in selected_cols if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric:
                raise ValueError(f"OutlierIQRHandler: Columns {non_numeric} are not numeric.")
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierIQRHandler':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)

        for col in self._processed_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)

        if self.action == 'cap':
            for col in self._processed_columns:
                if col in X_transformed.columns and col in self.bounds_: # Ensure col still exists
                    lower, upper = self.bounds_[col]
                    X_transformed[col] = np.clip(X_transformed[col], lower, upper)
        elif self.action == 'remove_rows':
            # Identify rows with outliers in ANY of the processed columns
            outlier_mask = pd.Series(False, index=X_transformed.index)
            for col in self._processed_columns:
                if col in X_transformed.columns and col in self.bounds_:
                    lower, upper = self.bounds_[col]
                    # Ensure comparison is valid (e.g., handle NaNs which are not outliers by this def)
                    col_values = X_transformed[col].dropna()
                    if not col_values.empty:
                         col_outliers = (X_transformed[col] < lower) | (X_transformed[col] > upper)
                         outlier_mask = outlier_mask | col_outliers.fillna(False) # NaNs are not outliers
            X_transformed = X_transformed[~outlier_mask]
        return X_transformed
    # get_feature_names_out from BaseTransformer is suitable.