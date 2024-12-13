from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from src.utils.data import download_data, make_preprocessed_edges_file, split_data, load_data, run_cleora, save_report, add_on_features
from src.utils.plots import visualize_pipeline, save_cm
from src.pipelines.data import DataPipeline
from src.pipelines.model import model_pipeline
from imblearn.metrics import geometric_mean_score
from src.config import config
import typer
from rich import print
import pandas as pd

def print_config():
    """
    Print the current configuration to the console.
    """
    print("Current Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

def main():
    print_config()

    # prepare embeddings
    embeddings_pipeline = DataPipeline([
        ("Download data", download_data),
        ("Preprocess edges", make_preprocessed_edges_file),
        ("Run Cleora", run_cleora)
    ])
    embeddings_pipeline.run()

    # prepare data for ML
    steps = [
    ("Download data", download_data),
    ("Split data", split_data),
    ("Load Data", load_data)
    ]
    if config['cleora_features_bool']:
        steps.append(("Add node features", add_on_features))

    data_pipeline = DataPipeline(steps)
    X_train, X_test, y_train, y_test = data_pipeline.run()
    visualize_pipeline(data_pipeline, type='html2')
    
    # prepare models
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=config['random_state'], class_weight="balanced"),
        "SGDClassifier1": SGDClassifier(random_state=config['random_state'], loss='log_loss', alpha=0.0001, penalty='elasticnet'),
        "LogisticRegression": LogisticRegression(random_state=config['random_state'], class_weight="balanced"),
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=config['random_state']),
        "KNeighbors": KNeighborsClassifier(n_neighbors=config['knn_neighbors'])
    }

    # train
    for model_name, model in models.items():
        print(f"\nTraining pipeline with {model_name}...")
        pipeline = model_pipeline(model)
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # evaluate model performance
        f1 = f1_score(y_test, y_pred, average='macro')
        g_mean = geometric_mean_score(y_test, y_pred)
        accuracy = pipeline.score(X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

        print(f"\n{model_name} Model Evaluation:")
        print(f"Macro F1: {f1:.4f}\nAccuracy: {accuracy:.4f}\nG-mean: {g_mean:.4f}")
        print(df.round(2).to_string())
        print("******************************\n")

        # save results
        save_report(df, model_name)
        cm = confusion_matrix(y_test, y_pred)
        save_cm(cm, model)
        
if __name__ == '__main__':
    typer.run(main)