# dataprep_lite/cleaning/text_cleaning.py
import pandas as pd
import numpy as np # Ensure numpy is imported
import re
from typing import List, Optional
from ..core.base_transformer import BaseTransformer
# from ..core.utils import identify_categorical_columns # Not strictly needed if using dtypes below

class BasicTextCleaner(BaseTransformer):
    """
    Performs basic text operations: lowercase, strip whitespace, remove punctuation.
    """
    def __init__(self,
                 columns_to_process: Optional[List[str]] = None,
                 lowercase: bool = True,
                 strip_whitespace: bool = True,
                 remove_punctuation: bool = True,
                 punctuation_regex: str = r'[^\w\s]'
                 ):
        super().__init__(columns_to_process)
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        # Compile regex only once if it's going to be used
        self.punctuation_regex_ = re.compile(punctuation_regex) if self.remove_punctuation and punctuation_regex else None
        self._original_dtypes: dict = {} # To store original dtypes of processed columns


    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            selected_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            if not selected_cols:
                raise ValueError("BasicTextCleaner: No text-like (object, string, category) columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            # Validate selected_cols are string-like
            for col in selected_cols:
                if not pd.api.types.is_string_dtype(X[col]) and \
                   not pd.api.types.is_object_dtype(X[col]) and \
                   not pd.api.types.is_categorical_dtype(X[col]):
                    print(f"Warning: BasicTextCleaner processing column '{col}' which is not object, string, or category. Results may vary.")
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BasicTextCleaner':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        # Store original dtypes of the columns that will be processed
        for col in self._processed_columns:
            if col in X.columns: # Ensure column exists in fit DataFrame
                self._original_dtypes[col] = X[col].dtype
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X) # X_transformed is a DataFrame copy

        for col in self._processed_columns:
            if col not in X_transformed.columns: continue

            # Get the original series from the current DataFrame to check its initial state in this transform call
            original_series_in_transform = X_transformed[col]
            is_na_mask = original_series_in_transform.isna()

            # Convert to Python string objects for universal operations, only non-NA values
            # This helps preserve the original NaN type if possible.
            temp_series_str_values = original_series_in_transform.dropna().astype(str)

            if self.lowercase:
                temp_series_str_values = temp_series_str_values.str.lower()
            if self.strip_whitespace:
                temp_series_str_values = temp_series_str_values.str.strip()
            if self.remove_punctuation and self.punctuation_regex_:
                temp_series_str_values = temp_series_str_values.str.replace(self.punctuation_regex_, '', regex=True)
            
            # Create a new series to assign back, preserving original index and NaN positions
            # Initialize with NaNs of the appropriate type
            original_dtype_name = self._original_dtypes.get(col, original_series_in_transform.dtype).name

            if original_dtype_name == 'string': # Check if original was pandas StringDtype
                # print(f"Column {col} was originally string, using pd.NA")
                nan_placeholder = pd.NA
                # Initialize with pd.NA, then fill. Resulting dtype should be StringDtype.
                new_col_series = pd.Series(nan_placeholder, index=original_series_in_transform.index, dtype=pd.StringDtype())
            else: # For object, category, etc., use np.nan initially
                # print(f"Column {col} was originally {original_dtype_name}, using np.nan")
                nan_placeholder = np.nan
                # Initialize with np.nan. Resulting dtype will likely be object.
                new_col_series = pd.Series(nan_placeholder, index=original_series_in_transform.index, dtype=object)

            # Update non-NA values
            new_col_series.update(temp_series_str_values)
            X_transformed[col] = new_col_series
            
            # If original was category, try to convert back if possible
            # This is tricky because cleaning might create values not in original categories
            if original_dtype_name == 'category':
                try:
                    # Only convert if all new values are in original categories or handle new ones
                    # For simplicity, if it was category, it might become object after cleaning
                    # A more advanced approach would be to update categories.
                    X_transformed[col] = X_transformed[col].astype('category')
                    # print(f"Column {col} successfully converted back to category.")
                except Exception as e:
                    # print(f"Could not convert {col} back to category: {e}. Remains object/string.")
                    pass


        return X_transformed