import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import VarianceThreshold
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_numeric_columns

class VarianceThresholdSelector(BaseTransformer):
    """Remove features with low variance."""

    def __init__(self, columns_to_process: Optional[List[str]] = None, threshold: float = 0.0):
        super().__init__(columns_to_process)
        self.threshold = threshold
        self.selector_: Optional[VarianceThreshold] = None

    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return identify_numeric_columns(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VarianceThresholdSelector':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        if not self._processed_columns:
            self.selector_ = None
            return self
        self.selector_ = VarianceThreshold(threshold=self.threshold)
        self.selector_.fit(X[self._processed_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = super().transform(X)
        if not self.selector_ or not self._processed_columns:
            return X_t
        current_cols = [c for c in self._processed_columns if c in X_t.columns]
        if not current_cols:
            return X_t
        reduced = self.selector_.transform(X_t[current_cols])
        kept_cols = [c for c, keep in zip(current_cols, self.selector_.get_support()) if keep]
        reduced_df = pd.DataFrame(reduced, columns=kept_cols, index=X_t.index)
        X_t = pd.concat([X_t.drop(columns=current_cols), reduced_df], axis=1)
        self._processed_columns = kept_cols
        return X_t

class CorrelationFilter(BaseTransformer):
    """Drop one of two features with correlation above threshold."""

    def __init__(self, columns_to_process: Optional[List[str]] = None, threshold: float = 0.95):
        super().__init__(columns_to_process)
        self.threshold = threshold
        self._columns_to_drop: List[str] = []

    def _get_compatible_columns(self, X: pd.DataFrame) -> List[str]:
        return identify_numeric_columns(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationFilter':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        corr = X[self._processed_columns].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self._columns_to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = super().transform(X)
        return X_t.drop(columns=[c for c in self._columns_to_drop if c in X_t.columns])
