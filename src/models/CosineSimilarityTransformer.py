from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class CosineSimilarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.train_embeddings = X
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        similarity_matrix = cosine_similarity(X, self.train_embeddings)
        return similarity_matrix