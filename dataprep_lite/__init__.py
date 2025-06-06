"""DataPrep-Lite: A lightweight data cleaning and preprocessing library."""

__version__ = "1.0.3"

from .core import (
    Pipeline,
    BaseTransformer,
    identify_numeric_columns,
    identify_categorical_columns,
    identify_datetime_columns,
    identify_boolean_columns,
)
from . import cleaning, preprocessing
from .cleaning import *
from .preprocessing import *
from .integration import to_pyarrow_table

__all__ = [
    "Pipeline",
    "BaseTransformer",
    "identify_numeric_columns",
    "identify_categorical_columns",
    "identify_datetime_columns",
    "identify_boolean_columns",
    "to_pyarrow_table",
] + cleaning.__all__ + preprocessing.__all__
