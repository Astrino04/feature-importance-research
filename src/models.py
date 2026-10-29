"""
Model Training Module

Trains three classifiers:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

All models use scikit-learn with default hyperparameters.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def train_models(X_train, y_train, random_state=42):
    """
    Train the three models on the training data.

    Parameters
    ----------
    X_train : ndarray
        Training features.
    y_train : ndarray
        Training labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    models : dict
        Dictionary mapping model name to fitted estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=10000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  Trained {name}")

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model on the test set and return accuracies.

    Parameters
    ----------
    models : dict
        Dictionary of fitted models.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test labels.

    Returns
    -------
    accuracies : dict
        Dictionary mapping model name to accuracy (float).
    """
    accuracies = {}
    print("\n=== Model Accuracies ===")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"  {name}: {acc:.1%}")

    return accuracies


if __name__ == "__main__":
    from data_loader import load_breast_cancer_data

    X_train, X_test, y_train, y_test, _ = load_breast_cancer_data()
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
