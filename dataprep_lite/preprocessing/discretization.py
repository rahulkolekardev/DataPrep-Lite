import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import KBinsDiscretizer
from ..core.base_transformer import BaseTransformer
from ..core.utils import identify_numeric_columns

class KBinsDiscretizerWrapper(BaseTransformer):
    """Wrapper for scikit-learn's KBinsDiscretizer."""
    def __init__(self,
                 columns_to_process: Optional[List[str]] = None,
                 n_bins: Union[int, List[int]] = 5,
                 encode: str = 'onehot-dense',  # 'onehot', 'onehot-dense', 'ordinal'
                 strategy: str = 'quantile',    # 'uniform', 'quantile', 'kmeans'
                 dtype: Optional[type] = None,
                 subsample: Optional[Union[int, str]] = 200_000, # For kmeans
                 random_state: Optional[int] = None): # For kmeans
        super().__init__(columns_to_process)
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
        self.discretizer_: Optional[KBinsDiscretizer] = None

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            selected_cols = identify_numeric_columns(X)
            if not selected_cols:
                raise ValueError("KBinsDiscretizerWrapper: No numeric columns found.")
        else:
            selected_cols = super()._get_columns_to_operate_on(X)
            non_numeric = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"KBinsDiscretizerWrapper: Columns {non_numeric} are not numeric.")
        return selected_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'KBinsDiscretizerWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)

        if not self._processed_columns:
            self.discretizer_ = None
            return self

        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins, encode=self.encode, strategy=self.strategy,
            dtype=self.dtype, subsample=self.subsample, random_state=self.random_state
        )
        # KBinsDiscretizer handles NaNs by assigning them to a separate bin if they exist,
        # or raises error if strategy cannot handle NaNs and they are present.
        # It's safer to impute NaNs before discretization if their handling is critical.
        self.discretizer_.fit(X[self._processed_columns].values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed_meta = super().transform(X)

        if not self.discretizer_ or not self._processed_columns:
            return X_transformed_meta
        
        # Ensure columns exist
        current_processed_cols = [c for c in self._processed_columns if c in X_transformed_meta.columns]
        if not current_processed_cols: return X_transformed_meta


        data_to_discretize = X_transformed_meta[current_processed_cols].values
        discretized_data = self.discretizer_.transform(data_to_discretize)

        # Get new feature names
        if hasattr(self.discretizer_, 'get_feature_names_out'):
            new_col_names = self.discretizer_.get_feature_names_out(current_processed_cols)
        else: # Fallback for older sklearn (less reliable for onehot)
            if self.encode == 'ordinal':
                new_col_names = [f"{col}_bin" for col in current_processed_cols]
            else: # 'onehot' or 'onehot-dense' - simplified naming
                new_col_names = []
                for i, col_name in enumerate(current_processed_cols):
                    n_bins_for_feat = self.discretizer_.n_bins_[i] # Number of bins for this feature
                    for bin_idx in range(int(n_bins_for_feat)):
                        new_col_names.append(f"{col_name}_bin_{bin_idx}")
        
        if self.encode == 'onehot' and hasattr(discretized_data, "toarray"): # Sparse output
            discretized_arr = discretized_data.toarray()
        else: # Dense output ('ordinal' or 'onehot-dense')
            discretized_arr = discretized_data
        
        discretized_df = pd.DataFrame(discretized_arr, columns=new_col_names, index=X_transformed_meta.index)

        X_others = X_transformed_meta.drop(columns=current_processed_cols)
        return pd.concat([X_others, discretized_df], axis=1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()
        if input_features is None:
            if self._feature_names_in is None: raise ValueError("Input features not known.")
            input_features_ = self._feature_names_in[:]
        else:
            input_features_ = list(input_features)

        if not self.discretizer_ or not self._processed_columns:
            return input_features_

        other_cols = [col for col in input_features_ if col not in self._processed_columns]
        
        if hasattr(self.discretizer_, 'get_feature_names_out'):
            discretized_new_cols = self.discretizer_.get_feature_names_out(self._processed_columns).tolist()
        else: # Fallback
            if self.encode == 'ordinal':
                discretized_new_cols = [f"{col}_bin" for col in self._processed_columns]
            else:
                discretized_new_cols = []
                for i, col_name in enumerate(self._processed_columns):
                    n_bins_for_feat = self.discretizer_.n_bins_[i]
                    for bin_idx in range(int(n_bins_for_feat)):
                         discretized_new_cols.append(f"{col_name}_bin_{bin_idx}")
        
        return other_cols + discretized_new_cols