from typing import Any
import pandas as pd


def to_pyarrow_table(df: pd.DataFrame) -> Any:
    """Convert a DataFrame to a PyArrow Table if pyarrow is installed."""
    try:
        import pyarrow as pa
    except ImportError as e:
        raise ImportError("pyarrow is required for this operation") from e
    return pa.Table.from_pandas(df)
