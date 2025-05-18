# dataprep_lite/core/pipeline.py
import pandas as pd
from typing import List, Tuple, Optional, Any 
from .base_transformer import BaseTransformer

class Pipeline:
    """
    Chains multiple transformers sequentially.
    """
    def __init__(self, steps: List[Tuple[str, BaseTransformer]]):
        """
        Args:
            steps: List of (name, transformer) tuples.
        """
        self.steps = steps
        self._validate_steps()
        self._is_fitted = False

    def _validate_steps(self):
        if not isinstance(self.steps, list):
            raise TypeError("Steps must be a list.")
        if not self.steps and isinstance(self.steps, list): # Allow empty pipeline for now
             return


        # Check structure and name types first
        for step in self.steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError("Each step must be a (name, transformer) tuple.")
            name, transformer = step # Unpack here for name validation
            if not isinstance(name, str):
                raise TypeError(f"Step name must be a string, but got {name} of type {type(name)}.")

        # Now that names are validated as strings, proceed with other checks
        names, transformers = zip(*self.steps)
        
        if len(set(names)) != len(names):
            raise ValueError("Transformer names in the pipeline must be unique.")
        
        for i, transformer in enumerate(transformers): # Use enumerate to get name if needed for error
            current_name = names[i]
            if not (hasattr(transformer, "fit") and \
                    hasattr(transformer, "transform") and \
                    hasattr(transformer, "fit_transform")):
                raise TypeError(
                    f"Transformer '{current_name}' (object: {transformer}) "
                    f"does not have required fit/transform/fit_transform methods."
                )

    @property
    def named_steps(self):
        return dict(self.steps)

    def __getitem__(self, key: Any) -> BaseTransformer:
        """Access transformers by index or name."""
        if isinstance(key, slice):
            # Return a new Pipeline instance with the sliced steps
            return self.__class__(self.steps[key])
        if isinstance(key, int):
            return self.steps[key][1]
        if isinstance(key, str):
            try:
                return self.named_steps[key]
            except KeyError:
                raise KeyError(f"No step found with name '{key}' in pipeline {list(self.named_steps.keys())}")
        raise TypeError(f"Cannot access Pipeline item with key {key} of type {type(key)}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Pipeline':
        """Fit all transformers in the pipeline."""
        Xt = X
        if not self.steps: # Handle empty pipeline
            self._is_fitted = True
            return self

        for i, (name, transformer) in enumerate(self.steps):
            if i < len(self.steps) - 1: # All but the last
                Xt = transformer.fit_transform(Xt, y)
            else: # Last transformer
                transformer.fit(Xt, y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transforms to the data, assuming pipeline is fitted."""
        if not self._is_fitted:
             raise RuntimeError("Pipeline has not been fitted. Call 'fit' before 'transform'.")
        Xt = X
        for name, transformer in self.steps:
            Xt = transformer.transform(Xt)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit all transformers, then transform the data."""
        Xt = X
        if not self.steps: # Handle empty pipeline
            self._is_fitted = True
            return Xt # Return original data if pipeline is empty

        for name, transformer in self.steps:
            Xt = transformer.fit_transform(Xt, y)
        self._is_fitted = True
        return Xt