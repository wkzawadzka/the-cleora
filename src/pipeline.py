from sklearn.pipeline import Pipeline

def create_pipeline(model):
    pipeline = Pipeline(steps=[
        ('classifier', model)
    ])

    return pipeline