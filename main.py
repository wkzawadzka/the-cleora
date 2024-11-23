from src.data_loader import load_data
from src.pipeline import create_pipeline
from src.Preprocessing import Preprocessing
from src.CleoraFacade import CleoraFacade
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from pathlib import Path

def main():
    preprocessing_singleton = Preprocessing()
    preprocessing_singleton.make_preprocessed_edges_file()
    cleora = CleoraFacade()
    cleora.run_cleora(Path.cwd() / "data" / "preprocessed_edges.txt")
    
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        "KNeighbors": KNeighborsClassifier(n_neighbors=7), 
        "DecisionTree": DecisionTreeClassifier() 
    }
    
    for model_name, model in models.items():
            print(f"\nTraining pipeline with {model_name}...")
            pipeline = create_pipeline(model)
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # evaluate model performance
            print(f"\n{model_name} Model Evaluation:")
            print(f"Accuracy: {pipeline.score(X_test, y_test):.4f}")
            print(classification_report(y_test, y_pred))
        
if __name__ == '__main__':
    main()