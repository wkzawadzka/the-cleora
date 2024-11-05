from data_loader import load_data
from pipeline import create_pipeline
from knn import KNeighborsClassifier
from tree import DecisionTreeClassifier

def main():

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