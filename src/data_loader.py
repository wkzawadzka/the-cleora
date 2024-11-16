import pandas as pd
from src.Preprocessing import Preprocessing
from src.CleoraFacade import CleoraFacade

def load_data():
    preprocess = Preprocessing()
    train = preprocess.train
    test = preprocess.test

    cleora = CleoraFacade()
    cleora.run_cleora("data/preprocessed_edges.txt")
    embeddings, dimension = cleora.load_embeddings("embeddings/emb__cluster_id__node.out")

    train = integrate_embeddings(train, embeddings)
    test = integrate_embeddings(test, embeddings)
    
    X_train = pd.DataFrame(train['embedding'].tolist())
    X_test = pd.DataFrame(test['embedding'].tolist())
    
    y_train = train['ml_target']
    y_test = test['ml_target']
    
    return X_train, X_test, y_train, y_test

def integrate_embeddings(df, embeddings):
    return df.merge(embeddings, left_on='id', right_on='node', how='left').drop(columns=['node'])