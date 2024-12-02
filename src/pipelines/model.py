from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from src.models.transformers import CosineSimilarityTransformer 
from sklearn.utils import estimator_html_repr
import matplotlib.pyplot as plt
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as Pipeline2
from src.config import config

def model_pipeline(model, verbose=True, imblearn=False):
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
        if imblearn:
            over = SMOTE(sampling_strategy = 0.55, random_state=config['random_state'])
            under = RandomUnderSampler(sampling_strategy = 0.7, random_state=config['random_state'])
            pipeline = Pipeline2([
                ('oversampling', over),
                ('undersampling', under),
                ('classifier', model)  
             ], verbose=verbose)
        else:
            pipeline = Pipeline([
                ('classifier', model)  
            ], verbose=verbose)
    

    diagram_path = Path.cwd() / "diagrams" / f"pipeline_{type(model).__name__}.html"
    diagram_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(estimator_html_repr(pipeline))

    return pipeline