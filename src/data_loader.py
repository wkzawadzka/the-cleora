from preprocessing import Preprocessing

def load_data():
    preprocess = Preprocessing()
    
    X_train, X_test = preprocess.train.drop(columns=['target']), preprocess.test.drop(columns=['target'])
    y_train, y_test = preprocess.train['target'], preprocess.test['target']
    
    return X_train, X_test, y_train, y_test