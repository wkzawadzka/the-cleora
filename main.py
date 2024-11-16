from src.data_loader import load_data
from src.pipeline import create_pipeline
from src.knn import KNeighborsClassifier
from src.tree import DecisionTreeClassifier
from src.Preprocessing import Preprocessing

def main():
    preprocessing_singleton = Preprocessing()
    preprocessing_singleton.make_preprocessed_edges_file()
    
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    
    for model_name, model in models.items():
        print(f"\nTraining pipeline with {model_name}...")
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        accuracy = pipeline.score(X_test, y_test)
        print(f"{model_name} Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()