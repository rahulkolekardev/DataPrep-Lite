from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Any

class BaseTransformer(ABC):
    """
    Abstract base class for all transformers.
    Inspired by scikit-learn's TransformerMixin.
    """
    def __init__(self, columns_to_process: Optional[List[str]] = None):
        """
        Args:
            columns_to_process: List of column names to process.
                If None, subclasses determine default column selection logic.
        """
        self.columns_to_process = columns_to_process
        self._is_fitted: bool = False
        self._feature_names_in: Optional[List[str]] = None # All columns in X during fit
        self._processed_columns: List[str] = [] # Actual columns selected for processing by fit

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call 'fit' before 'transform'."
            )

    def _validate_and_copy_df(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        return X.copy() # Work on a copy to prevent unintended modifications

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        """
        Determines the actual list of columns to process.
        Subclasses override this to define default column selection (if columns_to_process is None)
        and to validate compatibility of user-provided columns.
        """
        if self.columns_to_process is None:
            # Subclass-specific logic to select default columns (e.g., all numeric, all text)
            # This base implementation will assume all columns if not overridden AND no specific needs.
            # However, most transformers will need to override this for sensible defaults.
            selected_cols = X.columns.tolist()
        else:
            missing_cols = [col for col in self.columns_to_process if col not in X.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns {missing_cols} specified in 'columns_to_process' not found "
                    f"in input DataFrame. Available columns: {X.columns.tolist()}"
                )
            selected_cols = self.columns_to_process
        
        # Subclasses should further validate if selected_cols are of compatible types.
        return selected_cols

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        """
        Fit the transformer to data X.
        Learned parameters should be stored as attributes ending with an underscore.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X for fit must be a pandas DataFrame.")
        self._feature_names_in = X.columns.tolist()
        # _processed_columns will be set by the subclass's call to _get_columns_to_operate_on
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation to data X."""
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X for transform must be a pandas DataFrame.")

        # Check if all columns that were processed during fit are present in X for transform
        missing_processed_cols = [col for col in self._processed_columns if col not in X.columns]
        if missing_processed_cols:
            raise ValueError(
                f"Input DataFrame for transform is missing columns that were processed "
                f"during fit: {missing_processed_cols}. "
                f"Columns processed during fit: {self._processed_columns}. "
                f"Transform input columns: {X.columns.tolist()}"
            )
        
        X_transformed = self._validate_and_copy_df(X)
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.
        Transformers that add, remove, or rename columns MUST override this method.
        """
        self._check_is_fitted()
        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError("Input features not known. Call fit first or provide input_features.")
            return self._feature_names_in[:] # Return a copy
        
        # Basic check: ensure input_features match what was seen in fit, if applicable
        # This default assumes the transformer does not change column names or numbers.
        return list(input_features)