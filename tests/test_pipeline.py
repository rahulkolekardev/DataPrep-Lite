# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from typing import List 

from dataprep_lite.core import Pipeline, BaseTransformer
from dataprep_lite.cleaning import MeanImputer
from dataprep_lite.preprocessing import MinMaxScalerWrapper

class DummyTransformer(BaseTransformer):
    def __init__(self, add_val=0, columns_to_process=None):
        super().__init__(columns_to_process)
        self.add_val = add_val

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            self._processed_columns = X.columns.tolist()
            return X.columns.tolist()
        else:
            super()._get_columns_to_operate_on(X) # Base class check
            self._processed_columns = self.columns_to_process
            return self.columns_to_process

    def fit(self, X, y=None):
        super().fit(X,y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        return self

    def transform(self, X):
        X_transformed = super().transform(X)
        cols_to_apply = [col for col in self._processed_columns if col in X_transformed.columns]
        if cols_to_apply:
             X_transformed[cols_to_apply] = X_transformed[cols_to_apply] + self.add_val
        return X_transformed

@pytest.fixture
def sample_df_numeric():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 5.0] 
    })

def test_pipeline_simple(sample_df_numeric):
    pipeline = Pipeline([
        ('imputer_A', MeanImputer(columns_to_process=['A'])),
        ('imputer_B_for_scaler', MeanImputer(columns_to_process=['B'])),
        ('scaler_B', MinMaxScalerWrapper(columns_to_process=['B']))
    ])

    df_copy = sample_df_numeric.copy()
    transformed_df = pipeline.fit_transform(df_copy)

    assert not transformed_df['A'].isnull().any(), "Column A should have no NaNs after imputer"
    assert not transformed_df['B'].isnull().any(), "Column B should have no NaNs after its imputer"
    assert transformed_df['B'].min() >= 0 and transformed_df['B'].max() <= 1, "Column B should be scaled"
    assert 'A' in transformed_df.columns
    assert 'B' in transformed_df.columns

def test_pipeline_dummy_transformers():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    pipeline = Pipeline([
        ('add_one', DummyTransformer(add_val=1)), 
        ('add_ten_col1', DummyTransformer(add_val=10, columns_to_process=['col1']))
    ])
    transformed_df = pipeline.fit_transform(df.copy())

    expected_col1 = df['col1'] + 1 + 10
    expected_col2 = df['col2'] + 1

    pd.testing.assert_series_equal(transformed_df['col1'], expected_col1, check_dtype=False)
    pd.testing.assert_series_equal(transformed_df['col2'], expected_col2, check_dtype=False)

def test_pipeline_get_item(sample_df_numeric):
    imputer = MeanImputer()
    scaler = MinMaxScalerWrapper()
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    assert pipeline['imputer'] is imputer
    assert pipeline[1] is scaler
    with pytest.raises(KeyError): 
        _ = pipeline['non_existent_step']
    with pytest.raises(IndexError): 
        _ = pipeline[2]
    with pytest.raises(TypeError): 
        _ = pipeline[None]


def test_pipeline_unique_names():
    with pytest.raises(ValueError, match="Transformer names in the pipeline must be unique."):
        Pipeline([
            ('step1', DummyTransformer()),
            ('step1', DummyTransformer()) 
        ])

def test_pipeline_invalid_steps_structure():
    with pytest.raises(TypeError, match="Steps must be a list."):
        Pipeline("not a list") # type: ignore
    with pytest.raises(TypeError, match=r"Each step must be a \(name, transformer\) tuple."): 
        Pipeline([("name_only")]) # type: ignore
    
    # Test for non-string name, expecting specific message from Pipeline._validate_steps
    with pytest.raises(TypeError, match=r"Step name must be a string, but got 123 of type <class 'int'>\."):
        Pipeline([(123, DummyTransformer())]) # type: ignore

class NotATransformer: 
    pass
def test_pipeline_invalid_transformer_type():
    # Use a more robust regex that doesn't depend on the exact memory address
    match_str = r"Transformer 'step1' \(object: .*\) does not have required fit/transform/fit_transform methods\."
    with pytest.raises(TypeError, match=match_str):
        Pipeline([("step1", NotATransformer())]) # type: ignore

def test_pipeline_unfitted_transform():
    pipeline = Pipeline([('dummy', DummyTransformer())])
    df = pd.DataFrame({'A': [1,2,3]})
    with pytest.raises(RuntimeError, match="Pipeline has not been fitted. Call 'fit' before 'transform'."):
        pipeline.transform(df)