import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import os

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