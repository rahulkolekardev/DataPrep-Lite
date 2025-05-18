import pandas as pd
import numpy as np
from typing import List

def identify_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Identifies numeric (integer or float) columns."""
    return df.select_dtypes(include=np.number).columns.tolist()

def identify_categorical_columns(df: pd.DataFrame, 
                                 include_object: bool = True,
                                 cardinality_threshold: int = -1) -> List[str]:
    """
    Identifies categorical columns.
    Considers 'category' dtype and optionally 'object' or 'string' dtypes.
    """
    dtypes_to_include = ['category']
    if include_object:
        dtypes_to_include.extend(['object', 'string'])
        
    cat_cols = df.select_dtypes(include=dtypes_to_include).columns.tolist()

    if cardinality_threshold > 0:
        cat_cols = [col for col in cat_cols if df[col].nunique() <= cardinality_threshold]
    return cat_cols

def identify_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Identifies datetime columns."""
    return df.select_dtypes(include=[np.datetime64, 'datetime64[ns]']).columns.tolist()

def identify_boolean_columns(df: pd.DataFrame) -> List[str]:
    """Identifies boolean columns."""
    return df.select_dtypes(include=[bool, np.bool_]).columns.tolist()