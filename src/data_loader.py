import pandas as pd
from src.Preprocessing import Preprocessing
from src.CleoraFacade import CleoraFacade
from pathlib import Path

def load_data():
    preprocess = Preprocessing()
    train = preprocess.train
    test = preprocess.test

    cleora = CleoraFacade()
    embeddings_path = Path.cwd() / "data" / "embeddings" / "emb__cluster_id__node.out"
    embeddings, dimension = cleora.load_embeddings(embeddings_path)

    train = integrate_embeddings(train, embeddings)
    test = integrate_embeddings(test, embeddings)
    
    X_train = pd.DataFrame(train['embedding'].tolist())
    X_test = pd.DataFrame(test['embedding'].tolist())
    
    y_train = train['ml_target']
    y_test = test['ml_target']
    
    return X_train, X_test, y_train, y_test

def integrate_embeddings(df, embeddings):
    return df.merge(embeddings, left_on='id', right_on='node', how='left').drop(columns=['node'])