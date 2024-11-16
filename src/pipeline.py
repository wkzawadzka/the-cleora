from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from src.models.CosineSimilarityTransformer import CosineSimilarityTransformer

def create_pipeline():
    pipeline = Pipeline([
        ('cosine_similarity', CosineSimilarityTransformer()),
        ('knn', KNeighborsClassifier(n_neighbors=7))
    ])

    return pipeline