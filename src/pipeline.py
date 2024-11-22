from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from src.models.CosineSimilarityTransformer import CosineSimilarityTransformer 

def create_pipeline(model, verbose=True):
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
    
    return pipeline