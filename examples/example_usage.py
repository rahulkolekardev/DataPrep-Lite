# example_usage.py

import pandas as pd
import numpy as np
from typing import List, Optional, Any # For type hints

# --- Import from your dataprep-lite library ---
# Core components
from dataprep_lite.core import Pipeline, BaseTransformer # BaseTransformer for custom example

# Cleaning transformers
from dataprep_lite.cleaning import (
    MeanImputer, MedianImputer, ModeImputer, ConstantImputer, DropMissing,
    DropDuplicates,
    OutlierIQRHandler,
    TypeConverter,
    BasicTextCleaner
)

# Preprocessing transformers
from dataprep_lite.preprocessing import (
    OneHotEncoderWrapper, LabelEncoderWrapper,
    MinMaxScalerWrapper, StandardScalerWrapper,
    KBinsDiscretizerWrapper,
    DatetimeFeatureCreator
)

def print_df_info(df, title="DataFrame"):
    """Helper function to print DataFrame info."""
    print(f"\n--- {title} ---")
    if df is None:
        print("DataFrame is None")
        print("---------------------\n")
        return
    print(df.head())
    print("\nShape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("---------------------\n")

# --- 1. Create Sample DataFrame ---
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 2, 9, 10], # Duplicate ID 2
    'Age': [25, 30, np.nan, 22, 30, 35, 40, 30, 28, np.nan],
    'City': ['New York  ', 'London', 'Paris', 'New York  ', np.nan, 'Tokyo', 'london', 'London', 'Paris', 'Berlin'], # Note extra spaces and case variation
    'Experience': ['5', '10', '3', '2', '10', '12.5', '15', '10', '6', '8'], # String type
    'Salary': [70000, 250000, 65000, 50000, 90000, 110000, 120000, 250000, 80000, 75000], # Outlier
    'JoinDate': ['2020-01-15', '2019-05-20', '2021-02-10', '2020-08-01', '2018-11-05',
                 '2019-07-22', '2017-03-30', '2019-05-20', '2021-09-10', '2022-01-05'],
    'Feedback': ['Good Job!', 'Excellent work.', 'NEEDS IMPROVEMENT', 'Okay.', 'Very good.',
                 'Satisfactory', 'Outstanding!', 'Excellent work.', 'Good', 'Average'],
    'IsActive': [True, False, True, True, np.nan, True, False, False, True, True] # Boolean with NaN
}
df_original = pd.DataFrame(data)
print_df_info(df_original, "Original DataFrame")

# --- 2. Using Individual Transformers ---

# Make a copy for individual transformations
df_transformed_stepwise = df_original.copy()

# --- Cleaning ---
print(">>> Applying TypeConverter for 'Experience' and 'IsActive'")
type_converter = TypeConverter(type_mapping={'Experience': 'float64', 'IsActive': 'boolean'})
df_transformed_stepwise = type_converter.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After TypeConverter")

print(">>> Applying BasicTextCleaner for 'City' and 'Feedback'")
text_cleaner = BasicTextCleaner(columns_to_process=['City', 'Feedback'], remove_punctuation=False)
df_transformed_stepwise = text_cleaner.fit_transform(df_transformed_stepwise)
if 'City' in df_transformed_stepwise.columns:
    df_transformed_stepwise['City'] = df_transformed_stepwise['City'].astype(str).str.title() # Ensure str before .str
print_df_info(df_transformed_stepwise, "After BasicTextCleaner & City Title Case")

print(">>> Applying MeanImputer for 'Age'")
mean_imputer_age = MeanImputer(columns_to_process=['Age'])
df_transformed_stepwise = mean_imputer_age.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After MeanImputer (Age)")

print(">>> Applying ModeImputer for 'City'")
mode_imputer_city = ModeImputer(columns_to_process=['City'])
df_transformed_stepwise = mode_imputer_city.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After ModeImputer (City)")

print(">>> Applying ConstantImputer for 'IsActive'")
constant_imputer_active = ConstantImputer(fill_value=False, columns_to_process=['IsActive'])
df_transformed_stepwise = constant_imputer_active.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After ConstantImputer (IsActive)")


print(">>> Applying DropDuplicates based on 'ID', keeping first")
duplicate_dropper = DropDuplicates(subset=['ID'], keep='first')
df_temp_dups = df_original.copy()
df_temp_dups = duplicate_dropper.fit_transform(df_temp_dups)
print_df_info(df_temp_dups, "After DropDuplicates (ID) - Applied to a fresh copy")


print(">>> Applying OutlierIQRHandler for 'Salary' (capping)")
outlier_handler = OutlierIQRHandler(columns_to_process=['Salary'], action='cap', factor=1.5)
df_transformed_stepwise = outlier_handler.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After OutlierIQRHandler (Salary capped)")


# --- Preprocessing (continuing with df_transformed_stepwise) ---

print(">>> Applying DatetimeFeatureCreator for 'JoinDate'")
datetime_creator = DatetimeFeatureCreator(
    columns_to_process=['JoinDate'],
    features_to_extract=['year', 'month', 'dayofweek', 'is_weekend'],
    drop_original=True
)
df_transformed_stepwise = datetime_creator.fit_transform(df_transformed_stepwise)
print_df_info(df_transformed_stepwise, "After DatetimeFeatureCreator")


print(">>> Applying OneHotEncoderWrapper for 'City'")
ohe_city = OneHotEncoderWrapper(columns_to_process=['City'], drop='first', sparse_output=False)
if 'City' in df_transformed_stepwise.columns:
    df_transformed_stepwise = ohe_city.fit_transform(df_transformed_stepwise)
    print_df_info(df_transformed_stepwise, "After OneHotEncoderWrapper (City)")
else:
    print("Column 'City' not found for OHE, skipping.")


print(">>> Applying LabelEncoderWrapper for 'Feedback'")
label_encoder_feedback = LabelEncoderWrapper(columns_to_process=['Feedback'], unknown_value=-1)
if 'Feedback' in df_transformed_stepwise.columns:
    df_transformed_stepwise = label_encoder_feedback.fit_transform(df_transformed_stepwise)
    print_df_info(df_transformed_stepwise, "After LabelEncoderWrapper (Feedback)")
else:
    print("Column 'Feedback' not found for Label Encoding, skipping.")


print(">>> Applying MinMaxScalerWrapper for numeric features")
numeric_cols_for_scaling = ['Age', 'Experience', 'Salary'] # Add datetime parts if they exist
if 'JoinDate_year' in df_transformed_stepwise.columns: numeric_cols_for_scaling.append('JoinDate_year')
if 'JoinDate_month' in df_transformed_stepwise.columns: numeric_cols_for_scaling.append('JoinDate_month')
if 'JoinDate_dayofweek' in df_transformed_stepwise.columns: numeric_cols_for_scaling.append('JoinDate_dayofweek')
if 'JoinDate_is_weekend' in df_transformed_stepwise.columns: numeric_cols_for_scaling.append('JoinDate_is_weekend')

available_numeric_cols_for_scaling = [
    col for col in numeric_cols_for_scaling
    if col in df_transformed_stepwise.columns and pd.api.types.is_numeric_dtype(df_transformed_stepwise[col])
]

if available_numeric_cols_for_scaling:
    min_max_scaler = MinMaxScalerWrapper(columns_to_process=available_numeric_cols_for_scaling)
    df_transformed_stepwise = min_max_scaler.fit_transform(df_transformed_stepwise)
    print_df_info(df_transformed_stepwise, "After MinMaxScalerWrapper")
else:
    print("No suitable numeric columns found for scaling based on the predefined list.")


print(">>> Applying KBinsDiscretizerWrapper for 'Age'")
if 'Age' in df_transformed_stepwise.columns and pd.api.types.is_numeric_dtype(df_transformed_stepwise['Age']):
    # Make sure Age_bin doesn't already exist if re-running cells
    if 'Age_bin' in df_transformed_stepwise.columns:
        df_transformed_stepwise = df_transformed_stepwise.drop(columns=['Age_bin'])
    kbins = KBinsDiscretizerWrapper(columns_to_process=['Age'], n_bins=3, encode='ordinal', strategy='quantile')
    df_transformed_stepwise = kbins.fit_transform(df_transformed_stepwise)
    print_df_info(df_transformed_stepwise, "After KBinsDiscretizerWrapper (Age)")
else:
    print("Column 'Age' not found or not numeric for KBinsDiscretizer, skipping.")


# --- Define Custom Transformers Needed for the Pipeline ---
class TitleCaseTransformer(BaseTransformer):
    def __init__(self, columns_to_process: List[str]):
        super().__init__(columns_to_process)

    def _get_columns_to_operate_on(self, X:pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            raise ValueError("TitleCaseTransformer requires 'columns_to_process' to be specified.")
        selected_cols = super()._get_columns_to_operate_on(X) # Validates existence
        # Optionally, add further validation (e.g., are columns string-like?)
        return selected_cols

    def fit(self, X, y=None):
        super().fit(X,y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        return self

    def transform(self, X):
        X_transformed = super().transform(X)
        for col in self._processed_columns:
            if col in X_transformed.columns:
                # Ensure we operate on string representations
                is_na = X_transformed[col].isna()
                # Convert to string, title case, then put NaNs back
                # This handles mixed types in object columns or pandas StringDtype
                temp_col_str = X_transformed[col].astype(str).str.title()
                # Restore NaNs based on original mask
                temp_col_str[is_na] = X_transformed[col][is_na] # Assign original NaN/NaT
                X_transformed[col] = temp_col_str
        return X_transformed

class ColumnDropper(BaseTransformer):
    def __init__(self, columns_to_drop: List[str]):
        super().__init__()
        self.columns_to_drop = columns_to_drop
        self._dropped_columns: List[str] = []

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        self._processed_columns = [col for col in X.columns if col not in self.columns_to_drop]
        self._dropped_columns = [col for col in self.columns_to_drop if col in X.columns]
        return self._processed_columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ColumnDropper':
        super().fit(X, y)
        self._get_columns_to_operate_on(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = super().transform(X)
        cols_to_actually_drop = [col for col in self._dropped_columns if col in X_transformed.columns]
        return X_transformed.drop(columns=cols_to_actually_drop, errors='ignore')

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()
        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError("Input features not known for get_feature_names_out.")
            input_features_ = self._feature_names_in[:]
        else:
            input_features_ = list(input_features)
        return [col for col in input_features_ if col not in self._dropped_columns]


# --- 3. Building and Using a Pipeline ---
print("\n\n=== DEMONSTRATING PIPELINE USAGE ===")
df_pipeline_input = df_original.drop(columns=['ID'], errors='ignore').copy() # Remove ID beforehand

pipeline = Pipeline([
    ('type_converter', TypeConverter(type_mapping={'Experience': 'float64', 'IsActive': 'boolean'})),
    ('text_clean_city', BasicTextCleaner(columns_to_process=['City'])), # Cleans City (lowercase, strip)
    ('city_title_case', TitleCaseTransformer(columns_to_process=['City'])), # Title cases City
    ('city_imputer', ModeImputer(columns_to_process=['City'])),
    ('age_imputer', MeanImputer(columns_to_process=['Age'])),
    ('active_imputer', ConstantImputer(fill_value=False, columns_to_process=['IsActive'])),
    ('salary_outlier', OutlierIQRHandler(columns_to_process=['Salary'])),
    ('feedback_text_cleaner', BasicTextCleaner(columns_to_process=['Feedback'], remove_punctuation=True)),
    ('feedback_label_encoder', LabelEncoderWrapper(columns_to_process=['Feedback'])),
    ('datetime_feats', DatetimeFeatureCreator(columns_to_process=['JoinDate'], drop_original=True)),
    ('city_ohe', OneHotEncoderWrapper(columns_to_process=['City'], drop='first')),
    ('scaler', MinMaxScalerWrapper()) # Let it find all remaining numeric columns to scale
])

print_df_info(df_pipeline_input, "DataFrame before Pipeline")
df_processed_pipeline = pipeline.fit_transform(df_pipeline_input.copy())
print_df_info(df_processed_pipeline, "DataFrame after Pipeline processing")


# --- 4. Example of a Simple Custom Transformer (ColumnDropper already defined) ---
print("\n\n=== DEMONSTRATING CUSTOM TRANSFORMER (ColumnDropper) IN ISOLATION ===")
df_custom_test = df_original.copy()
column_dropper = ColumnDropper(columns_to_drop=['ID', 'Feedback'])
df_dropped = column_dropper.fit_transform(df_custom_test)
print_df_info(df_dropped, "After Custom ColumnDropper (Isolation)")

print("\n\n=== DEMONSTRATING CUSTOM TRANSFORMER (ColumnDropper) IN PIPELINE ===")
pipeline_with_custom_dropper = Pipeline([
    ('type_converter', TypeConverter(type_mapping={'Experience': 'float64'})),
    ('col_dropper', ColumnDropper(columns_to_drop=['ID', 'Feedback', 'JoinDate'])), # Drop these early
    ('age_imputer', MeanImputer(columns_to_process=['Age']))
    # Add more steps as needed on the remaining columns
])
df_processed_custom_pipeline = pipeline_with_custom_dropper.fit_transform(df_original.copy())
print_df_info(df_processed_custom_pipeline, "After Pipeline with CustomDropper")


print("\n\nExample usage script finished.")