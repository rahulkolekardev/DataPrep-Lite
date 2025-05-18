# dataprep_lite/cleaning/missing_values.py
import pandas as pd
import numpy as np
from typing import List, Optional, Any, Union
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_numeric_columns

class _BaseImputer(BaseTransformer):
    """Base class for imputers, not for direct use."""
    def __init__(self, columns_to_process: Optional[List[str]] = None):
        super().__init__(columns_to_process)
        self.imputation_values_: dict = {}

    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        raise NotImplementedError

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            compatible_cols = self._get_compatible_columns(X)
            if not compatible_cols:
                raise ValueError(f"{self.__class__.__name__}: No compatible columns found in DataFrame.")
            return compatible_cols
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> '_BaseImputer':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        for col in self._processed_columns:
            if col in X_transformed.columns and col in self.imputation_values_:
                X_transformed[col] = X_transformed[col].fillna(self.imputation_values_[col])
            elif X_transformed[col].isnull().any() and col not in self.imputation_values_:
                print(f"Warning: No imputation value for column '{col}'. NaNs may persist.")
        return X_transformed


class MeanImputer(_BaseImputer):
    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return identify_numeric_columns(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MeanImputer':
        super().fit(X,y)
        for col in self._processed_columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"MeanImputer: Column '{col}' is not numeric.")
            if X[col].isnull().all():
                self.imputation_values_[col] = np.nan
            else:
                self.imputation_values_[col] = X[col].mean()
        return self

class MedianImputer(_BaseImputer):
    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return identify_numeric_columns(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MedianImputer':
        super().fit(X, y)
        for col in self._processed_columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"MedianImputer: Column '{col}' is not numeric.")
            if X[col].isnull().all():
                self.imputation_values_[col] = np.nan
            else:
                self.imputation_values_[col] = X[col].median()
        return self

class ModeImputer(_BaseImputer):
    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return X.columns.tolist()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ModeImputer':
        super().fit(X, y)
        for col in self._processed_columns:
            modes = X[col].mode()
            if not modes.empty:
                self.imputation_values_[col] = modes[0]
            else:
                self.imputation_values_[col] = np.nan
        return self

class ConstantImputer(_BaseImputer):
    def __init__(self, fill_value: Any, columns_to_process: Optional[List[str]] = None):
        super().__init__(columns_to_process)
        self.fill_value = fill_value

    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return X.columns.tolist()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ConstantImputer':
        super().fit(X, y)
        for col in self._processed_columns:
            self.imputation_values_[col] = self.fill_value
        return self


class DropMissing(BaseTransformer):
    def __init__(self, axis: Union[int, str] = 0, how: Optional[str] = None,
                 thresh: Optional[int] = None, subset: Optional[List[str]] = None):
        super().__init__(columns_to_process=None)
        self.axis = axis
        self.subset = subset
        self._kept_columns_on_fit: Optional[List[str]] = None

        if how is not None and thresh is not None:
            raise ValueError("Cannot set both 'how' and 'thresh' arguments for DropMissing.")
        
        self.how = how
        self.thresh = thresh

        if self.thresh is not None:
            self.how = None # Prioritize thresh, 'how' becomes irrelevant for pandas call
        elif self.how is None and self.thresh is None: # If user explicitly sets both to None (or they default to None)
            self.how = 'any' # Default pandas behavior for dropna

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DropMissing':
        super().fit(X, y)
        self._processed_columns = X.columns.tolist()

        if self.axis == 1 or str(self.axis).lower() == 'columns':
            dropna_kwargs = {'axis': self.axis, 'subset': self.subset}
            if self.thresh is not None:
                dropna_kwargs['thresh'] = self.thresh
            elif self.how is not None:
                dropna_kwargs['how'] = self.how
            # If self.how is None here, it means thresh was set, or user set how=None (pandas default to 'any')
            
            df_temp_dropped = X.dropna(**dropna_kwargs)
            self._kept_columns_on_fit = df_temp_dropped.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        
        current_subset = self.subset
        if current_subset:
            current_subset = [col for col in self.subset if col in X_transformed.columns]
            if not current_subset and self.subset:
                print(f"Warning: All subset columns {self.subset} for DropMissing not found in transform input.")
                current_subset = None
        
        dropna_kwargs = {'axis': self.axis, 'subset': current_subset}
        if self.thresh is not None:
            dropna_kwargs['thresh'] = self.thresh
        elif self.how is not None:
            dropna_kwargs['how'] = self.how
            
        return X_transformed.dropna(**dropna_kwargs)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()
        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError("Input features not known. Call fit first or provide input_features.")
            input_features_ = self._feature_names_in[:]
        else:
            input_features_ = list(input_features)
        
        if (self.axis == 1 or str(self.axis).lower() == 'columns') and self._kept_columns_on_fit is not None:
            return self._kept_columns_on_fit[:]
        return list(input_features_)