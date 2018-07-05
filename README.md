# jsd-mlflow-examples
This is a collection of MLflow project examples that you can directly run with mlflow CLI commands on using python.

The goal provide you with a set of samples to get you quickly started on MLFlow.

## Keras MLP Binary Classification Model.

This is a simple Keras neural network model with three layers, one input, one hidden,
and one output layer. It's a simple linear model: `y=Mx`. Given a random numbers of X values,
it learns to predict it's Y value, from a training set.

This Multi-layer Perceptron (MLP) for binary classification model's sources have been modified from this [gist](https://gist.github.com/candlewill/552fa102352ccce42fd829ae26277d24). 
You can use other network models from this gist in a similar fashion to experiment. 

The arguments to run this simple MLP Keras network model are as follows:

* `--drop_rate`: Optional argument with a default value is `0.5`.
* `--input_dim  `: Input dimension. Default is `20`.
* `--bs`: dimension and size of the data. Default is `(1000, 20)`
* `--output`: Output to connected hidden layers. Default is `64`.
* `--train_batch_size`: Training batch size. Default is `128`
* `--epochs`: Number of epochs for training. Default is `20`.

To run the current program with just python and yet log all metrics, use
the following command:

`python keras/keras_nn_model.py`
`python keras/keras_nn_model.py --output=128 --epochs=10`
`python keras/keras_dnn/main_nn.py --output=128 --epochs=10`

It will log metrics and parameters in the `mlruns` directory. 

Alternatively, you can run using the `mlflow` command.

`mlflow run . -e keras/keras_nn_model -P --drop_rate=0.3 -P output=128`
The next two examples are from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).
While the Jupyter notebooks can be found [here](https://github.com/fchollet/deep-learning-with-python-notebooks), I have modified the code 
to tailor for use with MLflow. The description and experimentation remain the same, hence it fits well with using MLflow to experiments
various networks layers and suggested parameters to evaluate the model.

## Classifying Movie Reviews: a Keras binary classification example

This contains the code samples found in Chapter 3, Section 5 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 

----

Two-class classification, or binary classification, may be the most widely applied kind of machine learning problem. In this example, we 
will learn to classify movie reviews into "positive" reviews and "negative" reviews, just based on the text content of the reviews.

## Classifying Newswires: a multi-class Keras classification example

This contains the code samples found in Chapter 3, Section 5 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 

----

In the above model we saw how to classify vector inputs into two mutually exclusive classes using a densely-connected neural network. 
But what happens when you have more than two classes? 

In this section, we will build a network to classify Reuters newswires into 46 different mutually-exclusive topics. Since we have many 
classes, this problem is an instance of "multi-class classification", and since each data point should be classified into only one 
category, the problem is more specifically an instance of "single-label, multi-class classification". If each data point could have 
belonged to multiple categories (in our case, topics) then we would be facing a "multi-label, multi-class classification" problem.



