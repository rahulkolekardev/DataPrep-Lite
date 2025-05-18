import pandas as pd
from typing import List, Optional, Union
from ..core.base_transformer import BaseTransformer

class DropDuplicates(BaseTransformer):
    """Removes duplicate rows, similar to pandas.drop_duplicates()."""
    def __init__(self, subset: Optional[List[str]] = None,
                 keep: Union[str, bool] = 'first',
                 ignore_index: bool = False):
        super().__init__(columns_to_process=None)
        self.subset = subset
        self.keep = keep
        self.ignore_index = ignore_index

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DropDuplicates':
        super().fit(X, y)
        self._processed_columns = X.columns.tolist() # Uses all columns for context
        if self.subset:
            missing_cols = [col for col in self.subset if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Subset columns {missing_cols} for DropDuplicates not in DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        
        current_subset = self.subset
        if current_subset:
            current_subset = [col for col in self.subset if col in X_transformed.columns]
            if not current_subset and self.subset:
                print(f"Warning: All subset columns {self.subset} for DropDuplicates not found in transform input.")
                current_subset = None
        
        return X_transformed.drop_duplicates(subset=current_subset, keep=self.keep, ignore_index=self.ignore_index)

    # get_feature_names_out from BaseTransformer is suitable as columns aren't added/renamed.