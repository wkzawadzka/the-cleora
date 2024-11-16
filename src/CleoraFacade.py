import os
import subprocess
import pandas as pd

from src.Config import config

class CleoraFacade:
    def __init__(self, dimension=128, cleora_binary_filemane="cleora-v1.2.3-x86_64-pc-windows-msvc", iterations=1):
        cleora_dir = "cleora_binaries"
        self.cleora_binary_path = os.path.join(cleora_dir, cleora_binary_filemane)
        self.dimension = dimension
        self.iterations = iterations

    def run_cleora(self, input_file):
        output_dir = 'embeddings'
        os.makedirs(output_dir, exist_ok=True)
        cleora_command = [
            self.cleora_binary_path,
            '-c', 'transient::cluster_id node',
            '--input', input_file,
            '-o', output_dir,
            #'--seed', config['random_seed'],
            '--dimension', str(self.dimension),
            '-n', str(self.iterations)
        ]
        subprocess.run(cleora_command, check=True)
    
    def load_embeddings(self, filepath):
        with open(filepath, 'r') as file:
            first_line = file.readline().strip()
            num_edges, dimension = map(int, first_line.split(' '))
        
        embeddings = pd.read_csv(filepath, sep=' ', header=None, skiprows=1)
        embeddings = embeddings.drop(columns=[1])  # Drop the unneeded number of neighbours column
        embeddings.columns = ['node'] + [f'emb_{i}' for i in range(dimension)]

        embeddings['embedding'] = embeddings.apply(lambda row: row[1:].values, axis=1)
        embeddings = embeddings[['node', 'embedding']]
        
        return embeddings, dimension