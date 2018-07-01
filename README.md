# mlflow-examles
This is a collection of MLflow project examples that you can directly run with mlflow CLI commands on using python.

The goal provide you with a set of samples to get you quickly started on MLFlow.

## Keras Test Model.

This is a simple Keras neural network model with three layers, one input, one hidden,
and one output layer. It's a simple linear model: `y=Mx`. Given a random numbers of X values,
it learns to predict it's Y value, from a training set.

The arguments to run this simple Keras network model are as follows:

* `--drop_rate`: Optional argument with a default value is `0.5`.
* `--input_dim  `: Input dimension. Default is `20`.
* `--bs`: dimension and size of the data. Default is `(1000, 20)`
* `--output`: Output to connected hidden layers. Default is `64`.
* `--train_batch_size`: Training batch size. Default is `128`
* `--epochs`: Number of epochs for training. Default is `20`.

To run the current program with just python and yet log all metrics, use
the following command:

`python keras/keras_test.py`
`python keras/keras_test.py --output=128 --epochs=10`

It will log metrics and parameters in the `mlruns` directory. 

Alternatively, you can run using the `mlflow` command.

`mlflow run . e keras-test` -P --drop_rate=0.3 -P output=128`


