from src.data_loader import load_data
from src.pipeline import create_pipeline
from src.Preprocessing import Preprocessing
from src.CleoraFacade import CleoraFacade
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    preprocessing_singleton = Preprocessing()
    preprocessing_singleton.make_preprocessed_edges_file()
    
    iterations_range = range(1, 11)
    f1_scores = []

    model = DecisionTreeClassifier()
    model_name = model.__class__.__name__  
    
    for iterations in iterations_range:
        print(f"\nRunning experiment with Cleora iterations = {iterations}...")
        
        cleora = CleoraFacade(iterations=iterations)
        cleora.run_cleora(Path.cwd() / "data" / "preprocessed_edges.txt")
        
        X_train, X_test, y_train, y_test = load_data()
        
        pipeline = create_pipeline(model)
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        f1_scores.append(f1)
        
        print(f"F1-Score for {iterations} iterations: {f1:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_range, f1_scores, marker='o', linestyle='-', color='b', label='F1-Score')
    plt.title(f"F1-Score vs. Cleora Iterations ({model_name})") 
    plt.xlabel("Cleora Iterations")
    plt.ylabel("F1-Score")
    plt.xticks(iterations_range)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
