from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import pandas as pd
import mlflow
import argparse

# get argument from command line
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, required=False, default=3)
args = parser.parse_args()


if __name__ == "__main__":
    # Load and split data
    iris = load_iris()

    # create X (features) and y (response)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    n_neighbors = args.n_estimators

    exp = mlflow.set_experiment(experiment_name="iris_knn_experiment")

    tags = {"n_neighbors": n_neighbors, "data": "iris", "model": "knn",
            "framework": "sklearn", "purpose": "practice", "version": "1.0"}

    print("--- Experiment ---")
    print("Name: {}".format(exp.name))
    print("Experiment id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Created Time: {}".format(exp.creation_time))
    print("--------------------\n")

    mlflow.start_run()

    # Log tags
    mlflow.set_tags(tags)

    # Log data to mlflow
    pd_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataset = mlflow.data.from_pandas(pd_df)
    mlflow.log_input(dataset, context="testing")

    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    cross_val_scores = cross_val_score(
        model, X, y, cv=n_neighbors, scoring="accuracy")

    cv_mean = np.mean(cross_val_scores)

    print("Accuracy: ", accuracy)
    print("Cross Validation Mean: ", cv_mean)

    # Log parameters
    mlflow.log_param("n_neighbors", n_neighbors)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("cross_val_mean", cv_mean)

    # Log model
    mlflow.sklearn.log_model(model, "knn-model")

    mlflow.end_run()
