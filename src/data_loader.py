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
    
    X_train, X_test = preprocess.train['embedding'], preprocess.test.drop(columns=['ml_target'])
    y_train, y_test = preprocess.train['ml_target'], preprocess.test['ml_target']
    
    return X_train, X_test, y_train, y_test

def integrate_embeddings(df, embeddings):
    return df.merge(embeddings, left_on='id', right_on='node', how='left').drop(columns=['node'])