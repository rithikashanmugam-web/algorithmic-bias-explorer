from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


def mitigate_bias(preprocessor, X_train, y_train, sensitive_features):
    """
    Trains fairness-constrained model using Fairlearn
    """

    # Transform data using preprocessor
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Convert sparse → dense (Fairlearn needs dense)
    X_train_transformed = X_train_transformed.toarray()

    # Base model
    base_model = LogisticRegression(max_iter=1000)

    # Fair model
    fair_model = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity()
    )

    # Train with sensitive feature
    fair_model.fit(
        X_train_transformed,
        y_train,
        sensitive_features=sensitive_features
    )

    return fair_model
