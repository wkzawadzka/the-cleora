import requests
from zipfile import ZipFile
from pathlib import Path
from src.config import config
import pandas as pd
from sklearn.model_selection import train_test_split
from src.cleora import CleoraFacade

def make_preprocessed_edges_file(data=None):
    edges_df = pd.read_csv("data/git_web_ml/musae_git_edges.csv")

    with open("data/clique_edges.txt", "w") as cleora_clique, open("data/star_edges.txt", "w") as cleora_star:
        grouped_edges = edges_df.groupby('id_1')
        #for n, (name, group) in enumerate(grouped_train):
        for n, (name, group) in enumerate(grouped_edges):
            group_list = group['id_2'].tolist()
            group_elems = list(map(str, group_list))
            cleora_clique.write("{} {}\n".format(name, ' '.join(group_elems)))
            cleora_star.write("{}\t{}\n".format(n, name))
            for elem in group_elems:
                cleora_star.write("{}\t{}\n".format(n, elem))

def run_cleora(data=None):
    cleora = CleoraFacade(
        dimension=config['embedings_dimensions'],
        iterations=config['cleora_iterations']
    )
    cleora.run_cleora(config['cleora_expanison_type'])
    return cleora

def download_data(data=None):
    url = config['data_url']
    binaries_url = config['binaries_url']
    zip_path = Path("data") / "git_web_ml.zip"
    extract_dir = Path("data")
    binaries_dir = extract_dir / "cleora_binaries"

    # Check if the data is already downloaded
    if not (
        (extract_dir / "git_web_ml" / "musae_git_edges.csv").exists() and
        (extract_dir / "git_web_ml" / "musae_git_target.csv").exists() and
        (extract_dir / "git_web_ml" / "musae_git_features.json").exists() and
        (binaries_dir / binaries_url.split("/")[-1]).exists()
    ):
        # Ensure directories exist
        binaries_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset ZIP
        print("Downloading dataset...")
        response = requests.get(url)
        zip_path.write_bytes(response.content)

        # Download binaries
        print("Downloading binaries...")
        response = requests.get(binaries_url, stream=True)
        response.raise_for_status()
        output_file = binaries_dir / binaries_url.split("/")[-1]
        with output_file.open('wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract ZIP
        print("Extracting dataset...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up ZIP file
        zip_path.unlink()
        print("Download and extraction complete.")
    else:
        print("Data already exists.")

def split_data(data=None):
    df = pd.read_csv(Path("data") /  "git_web_ml" / "musae_git_target.csv")
    train, test = train_test_split(df, test_size=config['train_test_split'], random_state=config['random_seed'])
    print(train.shape)
    print(train.head())
    return train, test

def load_data(data=None):
    train, test = data
    cleora = CleoraFacade()
    if config['cleora_expanison_type'] == 'star':
        embeddings_path = Path.cwd() / "data" / "embeddings" / config['star_embedings_filename']
    elif config['cleora_expanison_type'] == 'clique':
        embeddings_path = Path.cwd() / "data" / "embeddings" / config['clique_embedings_filename']
    else:
        raise ValueError("Invalid expansion type. Choose either 'star' or 'clique'.")
    
    embeddings, dimension = cleora.load_embeddings(embeddings_path)

    train = integrate_embeddings(train, embeddings)
    test = integrate_embeddings(test, embeddings)
    
    X_train = pd.DataFrame(train['embedding'].tolist(), index=train.index)
    X_test = pd.DataFrame(test['embedding'].tolist(), index=test.index)
    
    y_train = train['ml_target']
    y_test = test['ml_target']
    
    return X_train, X_test, y_train, y_test

def integrate_embeddings(df, embeddings):
    merged_df = df.merge(embeddings, left_on='id', right_on='node', how='left')
    # set 'id' as the index
    merged_df = merged_df.set_index('node')
    return merged_df.drop(columns=['node'], errors='ignore')
