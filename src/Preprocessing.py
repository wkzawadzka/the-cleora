import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config
import os
import requests
from zipfile import ZipFile
from pathlib import Path

class Preprocessing():
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Preprocessing, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.download_data()
            df = pd.read_csv("data/git_web_ml/musae_git_target.csv")
            self.train, self.test = train_test_split(df, test_size=config['train_test_split'], random_state=config['random_seed'])
            print(self.train.shape)
            print(self.train.head())

    def download_data(self):
        url = config['data_url']
        binaries_url = config['binaries_url']
        zip_path = "data/git_web_ml.zip"
        extract_dir = "data"
        binaries_dir = Path.cwd() / extract_dir / "cleora_binaries"

        # Check if the data is already downloaded
        if not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_edges.csv")) or not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_target.csv")) or not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_features.json")):

            os.makedirs(binaries_dir, exist_ok=True)

            print("Downloading data...")
            response = requests.get(url)
            with open(zip_path, "wb") as file:
                file.write(response.content)

            response = requests.get(binaries_url, stream=True)
            response.raise_for_status()
            output_file = binaries_dir / "cleora-v1.2.3-x86_64-pc-windows-msvc"
            with open(output_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print("Extracting data...")
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # remove the zip file after extraction
            os.remove(zip_path)

    def make_preprocessed_edges_file(self):
            edges_df = pd.read_csv("data/git_web_ml/musae_git_edges.csv")

            with open("data/preprocessed_edges.txt", "w") as file:
                grouped_edges = edges_df.groupby('id_1')
                for n, (id_1, group) in enumerate(grouped_edges):
                    group_elems = group['id_2'].tolist()
                    file.write("{}\t{}\n".format(n, id_1))
                    for elem in group_elems:
                        file.write("{}\t{}\n".format(n, elem))

                        """
    def make_preprocessed_edges_file(self):
        edges_df = pd.read_csv("data/git_web_ml/musae_git_edges.csv")

        # Create reversed edges - cleora by default treats the graph as directed
        #reversed_edges_df = edges_df.rename(columns={"id_1": "id_2", "id_2": "id_1"})
        #edges_df = pd.concat([edges_df, reversed_edges_df]).drop_duplicates().reset_index(drop=True)

        with open("data/preprocessed_edges.txt", "w") as file:
            grouped_edges = edges_df.groupby('id_1')
            for _, (id_1, group) in enumerate(grouped_edges):
                group_list = group['id_2'].tolist()
                group_elems = list(map(str, group_list))
                for id_2 in group_elems:
                    file.write(f"{id_1}\t{id_2}\n")
"""