# dataprep_lite/preprocessing/encoding.py
import pandas as pd
import numpy as np
from typing import List, Optional, Any # <<<--- Corrected import
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_categorical_columns

class OneHotEncoderWrapper(BaseTransformer):
    """Wrapper for scikit-learn's OneHotEncoder."""
    def __init__(self, columns_to_process: Optional[List[str]] = None,
                 handle_unknown: str = 'ignore',
                 drop: Optional[str] = None, # e.g., 'first', 'if_binary'
                 sparse_output: bool = False): # For sklearn >= 1.2
        super().__init__(columns_to_process)
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse_output = sparse_output
        self.encoder_: Optional[OneHotEncoder] = None

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            # Default to object and category columns
            selected_cols = identify_categorical_columns(X, include_object=True)
            if not selected_cols:
                raise ValueError("OneHotEncoderWrapper: No categorical (object, category, string) columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            # Could add validation for categorical nature here
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OneHotEncoderWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)

        if not self._processed_columns: # No columns to encode
            self.encoder_ = None
            return self

        self.encoder_ = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            drop=self.drop,
            sparse_output=self.sparse_output
        )
        # Ensure data passed to sklearn encoder is suitable (e.g., handle NaNs if necessary before fit/transform)
        # sklearn OHE generally handles NaNs by creating a feature for them if not instructed otherwise.
        self.encoder_.fit(X[self._processed_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed_meta = super().transform(X) # Base checks and copy

        if not self.encoder_ or not self._processed_columns: # No encoder fitted or no columns to process
            return X_transformed_meta

        # Ensure all processed columns are still present
        current_processed_cols = [col for col in self._processed_columns if col in X_transformed_meta.columns]
        if not current_processed_cols:
             print(f"Warning: OHE - None of the original processed columns ({self._processed_columns}) found in transform input. Returning as is.")
             return X_transformed_meta
        if len(current_processed_cols) < len(self._processed_columns):
            missing_cols_in_transform = set(self._processed_columns) - set(current_processed_cols)
            print(f"Warning: OHE - Columns {missing_cols_in_transform} processed during fit not in transform input. Proceeding with available: {current_processed_cols}")


        X_to_encode = X_transformed_meta[current_processed_cols]
        X_others = X_transformed_meta.drop(columns=current_processed_cols)

        encoded_data = self.encoder_.transform(X_to_encode)

        if self.sparse_output and hasattr(encoded_data, "toarray"): # Is sparse
            encoded_arr = encoded_data.toarray()
        else: # Already dense or became dense
            encoded_arr = encoded_data

        # Get feature names for the encoded columns
        if hasattr(self.encoder_, 'get_feature_names_out'):
            ohe_output_names = self.encoder_.get_feature_names_out(current_processed_cols).tolist()
        else: # Fallback for older scikit-learn
            ohe_output_names = self.encoder_.get_feature_names(current_processed_cols).tolist()

        encoded_df = pd.DataFrame(encoded_arr, columns=ohe_output_names, index=X_transformed_meta.index)

        return pd.concat([X_others, encoded_df], axis=1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()

        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError("Input features not known for get_feature_names_out.")
            input_features_ = self._feature_names_in[:]
        else:
            input_features_ = list(input_features)

        if not self.encoder_ or not self._processed_columns:
            return input_features_

        # Columns not processed by OHE
        other_cols = [col for col in input_features_ if col not in self._processed_columns]

        # New columns from OHE
        # _processed_columns are the original names of columns fed to OHE
        if hasattr(self.encoder_, 'get_feature_names_out'):
            ohe_new_cols = self.encoder_.get_feature_names_out(self._processed_columns).tolist()
        else: # Fallback for older scikit-learn (less robust)
            ohe_new_cols = self.encoder_.get_feature_names(self._processed_columns).tolist()

        return other_cols + ohe_new_cols


class LabelEncoderWrapper(BaseTransformer):
    """Wrapper for scikit-learn's LabelEncoder, applied column-wise."""
    def __init__(self, columns_to_process: Optional[List[str]] = None,
                 unknown_value: Any = -1): # Value for unseen labels during transform
        super().__init__(columns_to_process)
        self.encoders_: dict = {} # {column_name: LabelEncoder_instance}
        self.unknown_value = unknown_value # e.g., -1 or np.nan

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            selected_cols = identify_categorical_columns(X, include_object=True)
            if not selected_cols:
                raise ValueError("LabelEncoderWrapper: No categorical columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            # Add validation if specified columns are suitable for label encoding
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LabelEncoderWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)

        for col in self._processed_columns:
            le = LabelEncoder()
            # Fit on unique, non-NaN values to define classes
            # Convert to string before fitting to handle mixed types gracefully and avoid errors.
            # NaNs are implicitly handled by dropna().
            non_na_values = X[col].dropna().astype(str)
            if not non_na_values.empty:
                 le.fit(non_na_values)
            # If all values were NaN, le.classes_ will be empty.
            self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)

        for col in self._processed_columns:
            if col not in X_transformed.columns or col not in self.encoders_:
                # Column might have been dropped or was not processed during fit
                continue

            le = self.encoders_[col]
            original_series = X_transformed[col]
            # Initialize transformed_series with a type that can hold unknown_value and NaNs
            output_dtype = float if pd.isna(self.unknown_value) or self.unknown_value is np.nan else type(self.unknown_value)
            # If unknown_value is int, but NaNs are present, need float.
            if original_series.isnull().any() and not (pd.isna(self.unknown_value) or self.unknown_value is np.nan):
                if isinstance(self.unknown_value, int): # If unknown is int, promote to float for NaNs
                    output_dtype = float

            transformed_series = pd.Series(index=original_series.index, dtype=output_dtype)

            # Handle NaNs first: they remain NaNs
            nan_mask = original_series.isna()
            transformed_series[nan_mask] = np.nan

            # Process non-NaN values
            non_na_original = original_series[~nan_mask].astype(str) # Convert to string for comparison with le.classes_

            if not non_na_original.empty:
                # Map known values
                # Create a mapping from class string to its integer encoding
                class_to_int_map = {cls_val: i for i, cls_val in enumerate(le.classes_)}

                # Apply mapping for known values
                known_mask = non_na_original.isin(le.classes_)
                if known_mask.any():
                    transformed_series.loc[non_na_original[known_mask].index] = non_na_original[known_mask].map(class_to_int_map)

                # Handle unknown values (not in le.classes_ among non-NaNs)
                unknown_mask_non_na = ~known_mask
                if unknown_mask_non_na.any():
                    transformed_series.loc[non_na_original[unknown_mask_non_na].index] = self.unknown_value
            
            X_transformed[col] = transformed_series
        return X_transformed
    # get_feature_names_out from BaseTransformer is suitable as column names don't change.