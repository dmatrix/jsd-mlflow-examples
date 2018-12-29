import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
sk_model = tree.DecisionTreeClassifier()
sk_model = sk_model.fit(iris.data, iris.target)

# set the artifact_path to location where
# experiment artifacts will be saved
# log model params

mlflow.set_experiment("v0.8.1-udf")
with mlflow.start_run():
    mlflow.log_param("criterion", sk_model.criterion)
    mlflow.log_param("splitter", sk_model.splitter)
    # log model
    #
    mlflow.sklearn.log_model(sk_model, "sk_models")
