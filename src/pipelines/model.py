from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from src.models.transformers import CosineSimilarityTransformer 
from sklearn.utils import estimator_html_repr
import matplotlib.pyplot as plt
from pathlib import Path

def model_pipeline(model, verbose=True):
    """
    This function creates a pipeline based on the model passed in.
    We are adding a custom transformer (e.g., CosineSimilarityTransformer) for Knn, 
    and using default for other models such as DecisionTree.
    
    Args:
        model (sklearn classifier): The model to be used in the pipeline.
    
    Returns:
        Pipeline: A sklearn pipeline with optional custom transformers and classifiers.
    """

    if isinstance(model, KNeighborsClassifier):
        pipeline = Pipeline([
            ('cosine_similarity', CosineSimilarityTransformer()),  
            ('knn', model)  # KNN classifier
        ], verbose=verbose)
    else: # default
        pipeline = Pipeline([
            ('classifier', model)  
        ], verbose=verbose)
    
    diagram_path = Path.cwd() / "diagrams" / f"pipeline_{type(model).__name__}.html"
    diagram_path.parent.mkdir(parents=True, exist_ok=True)

    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(estimator_html_repr(pipeline))

    return pipeline