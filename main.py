from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from src.utils.data import download_data, make_preprocessed_edges_file, split_data, load_data, run_cleora
from src.utils.plots import visualize_pipeline, save_cm
from src.pipelines.data import DataPipeline
from src.pipelines.model import model_pipeline
from src.config import config

def main():

    # prepare embeddings
    embeddings_pipeline = DataPipeline([
        ("Download data", download_data),
        ("Preprocess edges", make_preprocessed_edges_file),
        ("Run Cleora", run_cleora)
    ])
    embeddings_pipeline.run()

    # prepare data for ML
    data_pipeline = DataPipeline([
        ("Download data", download_data),
        ("Split data", split_data),
        ("Load Data", load_data)
    ])
    X_train, X_test, y_train, y_test = data_pipeline.run()
    visualize_pipeline(data_pipeline, type='html2')
    
    # prepare models
    models = {
        "KNeighbors": KNeighborsClassifier(n_neighbors=7), 
        "DecisionTree": DecisionTreeClassifier(random_state=config['random_state'], class_weight="balanced"),
        "SGDClassifier1": SGDClassifier(random_state=config['random_state'], loss='log_loss', alpha=0.0001, penalty='elasticnet'),
        "LogisticRegression": LogisticRegression(random_state=config['random_state'], class_weight="balanced")
    }
    
    for model_name, model in models.items():
        print(f"\nTraining pipeline with {model_name}...")
        pipeline = model_pipeline(model, imblearn=True)
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # evaluate model performance
        print(f"\n{model_name} Model Evaluation:")
        print(f"Accuracy: {pipeline.score(X_test, y_test):.4f}")
        print(classification_report(y_test, y_pred))
        print("******************************\n")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        save_cm(cm, model)
        
if __name__ == '__main__':
    main()