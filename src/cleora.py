import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import Literal 

from src.config import config

class CleoraFacade:
    def __init__(self, dimension=128, cleora_binary_filemane="cleora-v1.2.3-x86_64-pc-windows-msvc", iterations=1):

        cleora_dir = "cleora_binaries"
        self.cleora_binary_path = Path.cwd() / "data" / cleora_dir / cleora_binary_filemane
        self.dimension = dimension
        self.iterations = iterations

    def run_cleora(self, expansion_type: Literal['clique', 'star'] = 'star'):

        output_dir = Path.cwd() / "data" / "embeddings"
        os.makedirs(output_dir, exist_ok=True)

        if expansion_type == 'star':
            input_file = Path.cwd() / "data/star_edges.txt"
            cleora_command = [
                self.cleora_binary_path,
                '-c', 'transient::cluster_id node',
                '--input', Path.cwd() / input_file,
                '-o', output_dir,
                '--dimension', str(self.dimension),
                '-n', str(self.iterations)
            ]
        elif expansion_type == 'clique':
            input_file = Path.cwd() / "data/clique_edges.txt"
            cleora_command = [
                self.cleora_binary_path,
                '-c', 'complex::reflexive::node',
                '--input', Path.cwd() / input_file,
                '-o', output_dir,
                '--dimension', str(self.dimension),
                '-n', str(self.iterations)
            ]
        else:
            raise ValueError("Invalid expansion type. Choose either 'star' or 'clique'.")
        
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