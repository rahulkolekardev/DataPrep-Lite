import pandas as pd
from typing import List, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from ..core.base_transformer import BaseTransformer

class StopWordRemover(BaseTransformer):
    """Remove stop words from text columns."""

    def __init__(self, columns_to_process: Optional[List[str]] = None, stop_words: Optional[List[str]] = None):
        super().__init__(columns_to_process)
        self.stop_words = set(stop_words) if stop_words else ENGLISH_STOP_WORDS

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if not cols:
                raise ValueError("StopWordRemover: No text columns found.")
            return cols
        return super()._get_columns_to_operate_on(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StopWordRemover':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = super().transform(X)
        for col in self._processed_columns:
            if col not in X_t.columns:
                continue
            X_t[col] = X_t[col].astype(str).apply(
                lambda text: " ".join([w for w in text.split() if w.lower() not in self.stop_words]) if isinstance(text, str) else text
            )
        return X_t

class TfidfVectorizerWrapper(BaseTransformer):
    """Apply TF-IDF vectorization to text columns."""

    def __init__(self, columns_to_process: Optional[List[str]] = None, max_features: Optional[int] = None):
        super().__init__(columns_to_process)
        self.max_features = max_features
        self.vectorizers_: dict = {}

    def _get_columns_to_operate_on(self, X: pd.DataFrame) -> List[str]:
        if self.columns_to_process is None:
            cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if not cols:
                raise ValueError("TfidfVectorizerWrapper: No text columns found.")
            return cols
        return super()._get_columns_to_operate_on(X)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TfidfVectorizerWrapper':
        super().fit(X, y)
        self._processed_columns = self._get_columns_to_operate_on(X)
        for col in self._processed_columns:
            vec = TfidfVectorizer(max_features=self.max_features)
            vec.fit(X[col].astype(str).fillna(""))
            self.vectorizers_[col] = vec
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = super().transform(X)
        new_cols = []
        for col in self._processed_columns:
            if col not in X_t.columns or col not in self.vectorizers_:
                continue
            vec = self.vectorizers_[col]
            tfidf = vec.transform(X_t[col].astype(str).fillna(""))
            tfidf_df = pd.DataFrame(tfidf.toarray(),
                                    columns=[f"{col}_tfidf_{feat}" for feat in vec.get_feature_names_out()],
                                    index=X_t.index)
            X_t = pd.concat([X_t.drop(columns=[col]), tfidf_df], axis=1)
            new_cols.extend(tfidf_df.columns.tolist())
        self._processed_columns = new_cols
        return X_t
