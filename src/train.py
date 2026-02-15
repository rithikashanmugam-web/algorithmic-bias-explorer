from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def train_model(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    return model
