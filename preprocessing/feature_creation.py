import pandas as pd
from typing import List, Optional
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_datetime_columns

class DatetimeFeatureCreator(BaseTransformer):
    """Extracts features (year, month, day, etc.) from datetime columns."""
    def __init__(self,
                 columns_to_process: Optional[List[str]] = None,
                 features_to_extract: Optional[List[str]] = None, # e.g., ['year', 'month', 'dayofweek']
                 drop_original: bool = False):
        super().__init__(columns_to_process)
        self.features_to_extract = features_to_extract if features_to_extract else \
            ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'hour', 'minute', 'second']
        self.drop_original = drop_original
        self._datetime_columns_original: List[str] = [] # Columns confirmed as datetime during fit
        self._new_feature_names_map: dict = {} # {original_col: [new_feature_names]}

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            # Default to actual datetime columns + object columns that can be converted
            dt_cols = identify_datetime_columns(X)
            obj_cols = X.select_dtypes(include='object').columns
            
            convertible_obj_cols = []
            for col in obj_cols:
                if X[col].empty: continue
                try:
                    # Attempt conversion on a small, non-null sample
                    sample = X[col].dropna().iloc[:5]
                    if not sample.empty:
                        pd.to_datetime(sample, errors='raise')
                        convertible_obj_cols.append(col)
                except (ValueError, TypeError, AttributeError):
                    pass # Not easily convertible
            selected_cols = list(set(dt_cols + convertible_obj_cols))
            if not selected_cols:
                raise ValueError("DatetimeFeatureCreator: No datetime or convertible object columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            # Further validation could check if these columns are indeed datetime or convertible
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DatetimeFeatureCreator':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        
        self._datetime_columns_original = []
        self._new_feature_names_map = {}
        temp_X = X.copy() # For potential temporary type conversion

        for col in self._processed_columns:
            is_dt = False
            if pd.api.types.is_datetime64_any_dtype(temp_X[col]):
                is_dt = True
            else: # Try to convert for fitting purposes
                try:
                    temp_X[col] = pd.to_datetime(temp_X[col], errors='raise')
                    is_dt = True
                except (ValueError, TypeError):
                    print(f"Warning: Column '{col}' could not be converted to datetime during fit. Skipping.")
                    continue
            
            if is_dt:
                self._datetime_columns_original.append(col)
                self._new_feature_names_map[col] = []
                dt_accessor = temp_X[col].dt
                for feature in self.features_to_extract:
                    if hasattr(dt_accessor, feature):
                        self._new_feature_names_map[col].append(f"{col}_{feature}")
                    # Add custom features like 'is_weekend' if desired
                    elif feature == 'is_weekend':
                         self._new_feature_names_map[col].append(f"{col}_is_weekend")

        if not self._datetime_columns_original:
            print("Warning: DatetimeFeatureCreator found no processable datetime columns after fit.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)

        for col in self._datetime_columns_original: # Iterate only over columns confirmed during fit
            if col not in X_transformed.columns: continue

            # Ensure column is datetime for .dt accessor in transform
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                try:
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert '{col}' to datetime in transform. Skipping feature extraction for it.")
                    continue
            
            if X_transformed[col].isnull().all(): # All NaT after conversion
                # Create new feature columns with all NaNs/NaTs if original is all NaT
                for feature_name_suffix in self.features_to_extract:
                    new_col_name = f"{col}_{feature_name_suffix}"
                    if new_col_name in self._new_feature_names_map.get(col, []): # Check if this feature was planned
                        X_transformed[new_col_name] = pd.NaT if "time" in feature_name_suffix else np.nan
                continue


            dt_accessor = X_transformed[col].dt
            for feature in self.features_to_extract:
                new_col_name = f"{col}_{feature}"
                # Only create features that were identified as possible during fit
                if col in self._new_feature_names_map and new_col_name in self._new_feature_names_map[col]:
                    if hasattr(dt_accessor, feature):
                        X_transformed[new_col_name] = getattr(dt_accessor, feature)
                    elif feature == 'is_weekend':
                        X_transformed[new_col_name] = dt_accessor.dayofweek.isin([5, 6]).astype(int)
        
        if self.drop_original:
            cols_to_drop = [c for c in self._datetime_columns_original if c in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=cols_to_drop, errors='ignore')

        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()
        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError("Input features not known.")
            current_features = self._feature_names_in[:]
        else:
            current_features = list(input_features)

        output_features = []
        processed_originals_for_removal = set()

        for col_name in current_features:
            if col_name in self._new_feature_names_map: # This column was processed for datetime features
                if not self.drop_original:
                    output_features.append(col_name)
                output_features.extend(self._new_feature_names_map[col_name])
                processed_originals_for_removal.add(col_name)
            else: # Column was not processed for datetime features
                output_features.append(col_name)
        
        # Ensure no duplicates if original was not dropped and also a new feature name collided (unlikely)
        return pd.unique(output_features).tolist()