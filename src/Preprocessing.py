import pandas as pd
from sklearn.model_selection import train_test_split
from config import config
import os
import requests
from zipfile import ZipFile

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
            df = pd.read_csv("../data/git_web_ml/musae_git_target.csv")
            self.train, self.test = train_test_split(df, test_size=config['train_test_split'], random_state=config['random_seed'])

    def download_data(self):
        url = config['data_url']
        zip_path = "../data/git_web_ml.zip"
        extract_dir = "../data"

        # Check if the data is already downloaded
        if not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_edges.csv")) or not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_target.csv")) or not os.path.exists(os.path.join(extract_dir,
            "git_web_ml/musae_git_features.json")):

            os.makedirs(extract_dir, exist_ok=True)

            print("Downloading data...")
            response = requests.get(url)
            with open(zip_path, "wb") as file:
                file.write(response.content)

            print("Extracting data...")
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # remove the zip file after extraction
            os.remove(zip_path)

    def make_preprocessed_edges_file(self):
        edges_df = pd.read_csv("../data/git_web_ml/musae_git_edges.csv")
        with open("../data/preprocessed_edges.txt", "w") as file:
            grouped_edges = edges_df.groupby('id_1')
            for _, (id_1, group) in enumerate(grouped_edges):
                group_list = group['id_2'].tolist()
                group_elems = list(map(str, group_list))
                for id_2 in group_elems:
                    file.write(f"{id_1}\t{id_2}\n")