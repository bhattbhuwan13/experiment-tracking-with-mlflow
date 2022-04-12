from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import mlflow

def main():
    mlflow.sklearn.autolog()
    try:
        experiment_id = mlflow.create_experiment("real experiment")
    except:
        mlflow.set_experiment("real experiment")
        experiment = mlflow.get_experiment_by_name("real experiment")
        experiment_id = experiment.experiment_id

    print(f"The experiment id is {experiment_id}")
    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        clf.fit(iris.data, iris.target)


if __name__ == "__main__":
    main()
