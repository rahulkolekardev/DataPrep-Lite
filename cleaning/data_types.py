import pandas as pd
from typing import List, Optional, Dict, Any
from ..core.base_transformer import BaseTransformer

class TypeConverter(BaseTransformer):
    """Converts specified columns to new data types."""
    def __init__(self, type_mapping: Dict[str, Any], errors: str = 'raise'):
        """
        Args:
            type_mapping: Dict of {column_name: target_type}.
                          Target types can be standard pandas/numpy dtypes,
                          or special strings: 'to_numeric', 'to_datetime', 'infer_objects'.
            errors: How to handle conversion errors ('raise', 'coerce', 'ignore').
                    Behavior depends on the conversion function used.
        """
        # columns_to_process is implicitly defined by the keys of type_mapping
        super().__init__(columns_to_process=list(type_mapping.keys()))
        self.type_mapping = type_mapping
        self.errors = errors

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        # Ensures columns in type_mapping exist.
        return super()._get_columns_to_operate_on(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TypeConverter':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        # No parameters to learn, but validates columns.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)

        for col, target_type in self.type_mapping.items():
            if col not in X_transformed.columns: # Column might have been dropped by a prior step
                print(f"Warning: Column '{col}' for TypeConverter not found in input. Skipping.")
                continue
            if col not in self._processed_columns: # Should not happen if fit was called on compatible data
                 print(f"Warning: Column '{col}' was not in _processed_columns for TypeConverter. Skipping.")
                 continue


            current_series = X_transformed[col]
            try:
                if target_type == 'to_numeric':
                    X_transformed[col] = pd.to_numeric(current_series, errors=self.errors)
                elif target_type == 'to_datetime':
                    X_transformed[col] = pd.to_datetime(current_series, errors=self.errors)
                elif target_type == 'infer_objects':
                    # infer_objects returns a new Series; it doesn't take an errors argument.
                    X_transformed[col] = current_series.infer_objects()
                else:
                    # For astype, 'errors' param is only for specific dtypes like DatetimeTZDtype
                    # and was added more generally in later pandas versions.
                    # We'll assume 'raise' as the primary mode for astype.
                    if self.errors in ['coerce', 'ignore'] and target_type not in ['to_numeric', 'to_datetime']:
                        print(f"Warning: '{self.errors}' for astype on col '{col}' to '{target_type}' may not behave as pd.to_numeric/datetime.")
                    X_transformed[col] = current_series.astype(target_type) # errors='raise' implicitly for most types
            except Exception as e:
                if self.errors == 'raise':
                    raise ValueError(f"Error converting column '{col}' to '{target_type}': {e}")
                # For 'coerce' with astype, error means original data is kept if astype fails.
                # For 'ignore' with astype, error means original data is kept.
                # pd.to_numeric/datetime handle 'coerce' by setting NaNs/NaTs.
                print(f"Warning: Failed to convert column '{col}' to '{target_type}' with errors='{self.errors}'. Error: {e}")
        return X_transformed
    # get_feature_names_out from BaseTransformer is suitable.