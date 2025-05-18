# dataprep_lite_project/tests/cleaning/test_missing_values.py
import pytest
import pandas as pd
import numpy as np

# Ensure your library can be imported
from dataprep_lite.cleaning.missing_values import (
    MeanImputer, MedianImputer, ModeImputer, ConstantImputer, DropMissing
)
from dataprep_lite.core.base_transformer import BaseTransformer # For testing inheritance if needed

@pytest.fixture
def df_with_nans():
    return pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
        'categorical_col': ['A', 'B', 'A', np.nan, 'B', 'B'],
        'all_nan_col': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'no_nan_col': [1, 2, 3, 4, 5, 6],
        'mixed_type_col_for_mode': [1, 'A', 1, np.nan, 'A', 1] # For mode imputer testing
    })

def test_mean_imputer_default(df_with_nans):
    imputer = MeanImputer() # Should process only 'numeric_col' and 'no_nan_col' by default
    df_transformed = imputer.fit_transform(df_with_nans.copy())

    assert df_transformed['numeric_col'].isnull().sum() == 0
    expected_mean_numeric = df_with_nans['numeric_col'].mean()
    assert df_transformed['numeric_col'].iloc[2] == expected_mean_numeric
    assert df_transformed['numeric_col'].iloc[5] == expected_mean_numeric

    # Non-numeric or all_nan columns should be largely untouched by default MeanImputer logic
    pd.testing.assert_series_equal(df_transformed['categorical_col'], df_with_nans['categorical_col'])
    assert df_transformed['all_nan_col'].isnull().all() # Mean of all NaNs is NaN, so fillna(NaN) has no effect
    assert not df_transformed['no_nan_col'].isnull().any() # Should remain unchanged

def test_mean_imputer_specific_col(df_with_nans):
    imputer = MeanImputer(columns_to_process=['numeric_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['numeric_col'].isnull().sum() == 0
    assert 'categorical_col' in df_transformed # Other columns should still exist
    assert df_transformed['categorical_col'].isnull().sum() == 1 # Unchanged

def test_median_imputer(df_with_nans):
    imputer = MedianImputer(columns_to_process=['numeric_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['numeric_col'].isnull().sum() == 0
    expected_median_numeric = df_with_nans['numeric_col'].median()
    assert df_transformed['numeric_col'].iloc[2] == expected_median_numeric

def test_mode_imputer_categorical(df_with_nans):
    imputer = ModeImputer(columns_to_process=['categorical_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['categorical_col'].isnull().sum() == 0
    # Mode of ['A', 'B', 'A', nan, 'B', 'B'] is 'B' (pandas behavior)
    assert df_transformed['categorical_col'].iloc[3] == 'B'

def test_mode_imputer_mixed_type(df_with_nans):
    # Mode of [1, 'A', 1, nan, 'A', 1] -> 1 is mode (appears 3 times)
    imputer = ModeImputer(columns_to_process=['mixed_type_col_for_mode'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['mixed_type_col_for_mode'].isnull().sum() == 0
    # The mode should be 1. Pandas converts to object if mixed, mode() returns object.
    assert df_transformed['mixed_type_col_for_mode'].iloc[3] == 1


def test_mode_imputer_all_cols_default(df_with_nans):
    imputer = ModeImputer() # Apply to all possible columns by default
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['numeric_col'].iloc[2] == df_with_nans['numeric_col'].mode()[0]
    assert df_transformed['categorical_col'].iloc[3] == 'B'
    assert df_transformed['all_nan_col'].isnull().all() # Mode of all NaNs is empty -> NaN imputed
    assert df_transformed['mixed_type_col_for_mode'].iloc[3] == 1

def test_constant_imputer_numeric(df_with_nans):
    imputer = ConstantImputer(fill_value=-99, columns_to_process=['numeric_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['numeric_col'].isnull().sum() == 0
    assert df_transformed['numeric_col'].iloc[2] == -99

def test_constant_imputer_string(df_with_nans):
    imputer = ConstantImputer(fill_value='MISSING', columns_to_process=['categorical_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['categorical_col'].isnull().sum() == 0
    assert df_transformed['categorical_col'].iloc[3] == 'MISSING'


def test_drop_missing_rows_any_subset(df_with_nans):
    # Drops rows if EITHER 'numeric_col' OR 'categorical_col' is NaN.
    # Original index: 0  1  2    3    4  5
    # numeric_col:    1  2 NaN   4   5 NaN
    # categorical_col:A  B  A   NaN  B  B
    # Row 2 (index 2) has NaN in numeric_col -> drop
    # Row 3 (index 3) has NaN in categorical_col -> drop
    # Row 5 (index 5) has NaN in numeric_col -> drop
    # Expected kept indices: 0, 1, 4
    dropper_subset = DropMissing(axis=0, how='any', subset=['numeric_col', 'categorical_col'])
    df_transformed_subset = dropper_subset.fit_transform(df_with_nans.copy())
    assert len(df_transformed_subset) == 3
    expected_indices = pd.Index([0, 1, 4])
    pd.testing.assert_index_equal(df_transformed_subset.index, expected_indices)
    # NaNs in other columns (like 'all_nan_col') of kept rows are preserved.
    assert df_transformed_subset['all_nan_col'].isnull().sum() == 3


def test_drop_missing_rows_all_original_df(df_with_nans):
    # Drops rows if ALL values in that row are NaN. No such row in df_with_nans.
    dropper = DropMissing(axis=0, how='all')
    df_transformed = dropper.fit_transform(df_with_nans.copy())
    assert len(df_transformed) == len(df_with_nans) # No rows should be dropped

def test_drop_missing_cols_all(df_with_nans):
    # Drops columns if ALL values in that column are NaN. 'all_nan_col' should be dropped.
    dropper = DropMissing(axis=1, how='all')
    df_transformed = dropper.fit_transform(df_with_nans.copy())
    assert 'all_nan_col' not in df_transformed.columns
    assert 'numeric_col' in df_transformed.columns # Should remain
    assert 'no_nan_col' in df_transformed.columns  # Should remain

def test_drop_missing_thresh(df_with_nans):
    # Keep rows with at least 4 non-NaN values
    # Row 0: 3 non-NaN (numeric, cat, no_nan) -> drop if all_nan_col counts as NaN
    # Let's test with subset for clarity
    df_subset_for_thresh = df_with_nans[['numeric_col', 'categorical_col', 'no_nan_col']].copy()
    # numeric_col: [1.0, 2.0, nan, 4.0, 5.0, nan]
    # cat_col:     [ A ,  B ,  A , nan,  B ,  B ]
    # no_nan_col:  [ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ]
    # Non-NaN counts per row for these 3 cols:
    # Row 0: 3
    # Row 1: 3
    # Row 2: 2 (A, 3)
    # Row 3: 2 (4.0, 4)
    # Row 4: 3
    # Row 5: 2 (B, 6)
    dropper = DropMissing(thresh=3, subset=['numeric_col', 'categorical_col', 'no_nan_col'])
    df_transformed = dropper.fit_transform(df_subset_for_thresh) # Pass the subsetted df
    assert len(df_transformed) == 3 # Rows 0, 1, 4
    pd.testing.assert_index_equal(df_transformed.index, pd.Index([0, 1, 4]))


def test_imputer_no_compatible_columns_mean():
    df_all_object = pd.DataFrame({'A': ['x', 'y', 'z'], 'B': ['p', 'q', 'r']})
    with pytest.raises(ValueError, match="MeanImputer: No compatible columns found in DataFrame."):
        MeanImputer().fit(df_all_object.copy())

def test_imputer_no_compatible_columns_median():
    df_all_object = pd.DataFrame({'A': ['x', 'y', 'z'], 'B': ['p', 'q', 'r']})
    with pytest.raises(ValueError, match="MedianImputer: No compatible columns found in DataFrame."):
        MedianImputer().fit(df_all_object.copy())


def test_constant_imputer_no_nans_does_nothing(df_with_nans):
    # ConstantImputer should work on any column type by default if no columns are specified
    imputer = ConstantImputer(fill_value="test", columns_to_process=['no_nan_col'])
    df_copy = df_with_nans.copy()
    transformed_df = imputer.fit_transform(df_copy)
    pd.testing.assert_series_equal(transformed_df['no_nan_col'], df_with_nans['no_nan_col']) # No NaNs to fill

def test_imputer_column_not_found_in_user_spec(df_with_nans):
    imputer = MeanImputer(columns_to_process=['NonExistentColumn'])
    with pytest.raises(ValueError, match="Columns \\['NonExistentColumn'\\] specified in 'columns_to_process' not found"):
        imputer.fit(df_with_nans.copy())

def test_mean_imputer_non_numeric_specified_by_user(df_with_nans):
    imputer = MeanImputer(columns_to_process=['categorical_col'])
    with pytest.raises(ValueError, match="MeanImputer: Column 'categorical_col' is not numeric."):
        imputer.fit(df_with_nans.copy())

def test_imputer_unfitted_transform():
    imputer = MeanImputer()
    df = pd.DataFrame({'A': [1, np.nan]})
    with pytest.raises(RuntimeError, match="MeanImputer has not been fitted. Call 'fit' before 'transform'."):
        imputer.transform(df)

def test_imputer_all_nans_column_mean(df_with_nans):
    imputer = MeanImputer(columns_to_process=['all_nan_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['all_nan_col'].isnull().all() # Mean of all NaNs is NaN, fillna(NaN) is no-op

def test_imputer_all_nans_column_median(df_with_nans):
    imputer = MedianImputer(columns_to_process=['all_nan_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['all_nan_col'].isnull().all() # Median of all NaNs is NaN

def test_imputer_all_nans_column_mode(df_with_nans):
    imputer = ModeImputer(columns_to_process=['all_nan_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert df_transformed['all_nan_col'].isnull().all() # Mode of all NaNs is empty Series -> impute NaN

def test_imputer_all_nans_column_constant(df_with_nans):
    imputer = ConstantImputer(fill_value=0, columns_to_process=['all_nan_col'])
    df_transformed = imputer.fit_transform(df_with_nans.copy())
    assert not df_transformed['all_nan_col'].isnull().any()
    assert (df_transformed['all_nan_col'] == 0).all()