import mlflow
import mlflow.tracking
import time

print(mlflow.__version__)

def get_clnt():

    clnt = mlflow.tracking.MlflowClient()
    return clnt

def get_exp_info(clnt):

    exp_name = "0.8.0_exp_" + str(int(time.time()))
    exp_id = clnt.create_experiment(exp_name)
    run_info = clnt.create_run(exp_id)
    run_id = run_info.info.run_uuid

    return [exp_id, run_id, exp_name]

def run_named_experiment():

    mlflow.set_experiment("0.8.0_test_nested")
    with mlflow.start_run(nested=True):
        mlflow.log_param("mse", 0.10)
        mlflow.log_param("lr", 0.05)
        mlflow.log_param("batch_size", 512)
        with mlflow.start_run(nested=True):
            mlflow.log_param("max_runs", 32)
            mlflow.log_param("epochs", 20)
            mlflow.log_metric("acc", 98)
            mlflow.log_metric("rmse", 98)
            with mlflow.start_run(nested=True):
                mlflow.log_param("max_runs", 60)
                mlflow.log_metric("mae", 96)

def run_default_experiment():

    with mlflow.start_run(nested=True):
        mlflow.log_param("mse", 0.10)
        mlflow.log_param("lr", 0.05)
        mlflow.log_param("batch_size", 512)
        with mlflow.start_run(nested=True):
            mlflow.log_param("max_runs", 32)
            mlflow.log_param("epochs", 20)
            mlflow.log_metric("acc", 98)
            mlflow.log_metric("rmse", 98)
            with mlflow.start_run(nested=True):
                mlflow.log_param("max_runs", 60)
                mlflow.log_metric("mae", 96)

if __name__ == '__main__':

    # run default experiment
    print("Running Nested Default Experiment...")
    run_default_experiment()
    # run named experiment
    print("Running Nested Named Experiment...")
    run_named_experiment()
