config = {
    'data_url': 'https://snap.stanford.edu/data/git_web_ml.zip',
    'binaries_url': 'https://github.com/BaseModelAI/cleora/releases/download/v1.2.3/cleora-v1.2.3-x86_64-pc-windows-msvc',
    'random_state': 2137,
    'train_test_split': 0.2,
    'random_seed': 2137,
    'star_embedings_filename': 'emb__cluster_id__node.out', # used in load_data() function in utils.data.py
    'clique_embedings_filename': 'emb__node__node.out', # used in load_data() function in utils.data.py

    # used in cleora.py and utils.data.py
    'embedings_dimensions': 128,

    # imblearn params
    'over_sampling_strategy': 0.55, #SMOTE
    'under_sampling_strategy': 0.7, 

    # experiments
    'experiment_name': 'it8_star_wofeat_imblearn_6',
    'cleora_iterations': 8,
    'cleora_expanison_type': 'star',
    'cleora_features_bool': False,
    'imblearn': True,
    'knn_neighbors': 6
}