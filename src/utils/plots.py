import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.utils import estimator_html_repr
from sklearn.metrics import ConfusionMatrixDisplay

def plot_class_distribution(target_df, output_dir='data/plots', filename='class_distribution.png'):
    """
    Creates a bar plot showing the class distribution of GitHub users
    and saves it as a PNG file.

    Parameters:
    - target_df: DataFrame containing the target classes.
    - output_dir: Directory to save the plot.
    - filename: Name of the output PNG file.
    """
    class_counts = target_df['ml_target'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Label'] = class_counts['Class'].map({0: 'Web devs', 1: 'ML Eng'})
    print(class_counts)

    colors = ['lightblue', 'lightgreen']

    plot = (
        ggplot(class_counts, aes(x='Label', y='Count')) +
        geom_bar(stat='identity', fill=colors) +
        labs(
            title='Class Distribution',
            x='Class',
            y='Number of Users'
        ) +
        theme_minimal() +
        # geom_text(aes(label='Count'), va='bottom', size=10, 
        #           data=class_counts, 
        #           position=position_stack(vjust=0.8)) + 
        scale_y_continuous(breaks=range(0, class_counts['Count'].max() + 5000, 5000)) +
        theme(
            figure_size=(5, 4),
            axis_text_x=element_text(rotation=0, ha='center')
        ) + 
        coord_flip()
    )

    # saving
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    ggsave(plot, filename=output_file, dpi=300)

    return(plot)


def visualize_pipeline(pipeline, output_path=Path("diagrams") / "data_pipeline.png", type='graph'):
    diagrams_dir = Path.cwd() / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)  # ensure 'diagrams' directory exists
    # as G ***************************
    if type == 'graph':
        G = nx.DiGraph()
        for i in range(len(pipeline.steps) - 1): 
            step_name = pipeline.steps[i][0]
            next_step_name = pipeline.steps[i + 1][0]
            
            G.add_node(step_name) 
            G.add_node(next_step_name)  
            G.add_edge(step_name, next_step_name) 
        plt.figure(figsize=(10, 8)) 
        pos = nx.spring_layout(G, seed=42)  
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)  
        plt.savefig(output_path, format="png")
        plt.close()  
        print(f"Pipeline diagram saved to {output_path}")

    elif type == 'html':
    # html ******************************
        diagram_path = Path.cwd() / "diagrams" / f"data_pipeline.html"
        diagram_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagram_path, "w", encoding="utf-8") as f:
            f.write(pipeline.to_html())
        print(f"Pipeline diagram saved to {diagram_path}")
    
    elif type == 'html2':
        # html ******************************
        diagram_path = Path.cwd() / "diagrams" / f"data_pipeline.html"
        diagram_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagram_path, "w", encoding="utf-8") as f:
            f.write(pipeline.to_html2())
        print(f"Pipeline diagram saved to {diagram_path}")

def save_cm(confusion_matrix, model, classes=["Web Devs", "ML Engs"]):
    """ Save a confusion matrix plot for a given model.

    Parameters:
        confusion_matrix (array-like): The confusion matrix to display.
        model (object): The model object (used to name the file).
        classes (list): List of class labels for the confusion matrix. """
    plt.title(f"Confusion Matrix - {type(model).__name__}")
    plt.tight_layout()

    diagrams_dir = Path.cwd() / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)  # ensure 'diagrams' directory exists
    diagram_path = diagrams_dir / f"cm_{type(model).__name__}.png"

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=classes)
    disp.plot()
    plt.savefig(diagram_path)
    plt.close() 

    print(f"Confusion matrix saved at: {diagram_path}")
