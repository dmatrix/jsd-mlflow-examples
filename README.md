# jsd-mlflow-examples
This is a collection of MLflow project examples that you can directly run with mlflow CLI commands or directly using Python.

The goal is provide you with additional set of samples, focusing on machine learning and deep learning examples, to get you quickly started on MLFlow. 
In particular, I focus on [Keras](https://keras.io) and borrow and extend examples
from Francois Cholllet's book: [_Deep Learning with Python_](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff)

## 1. Keras MLP Binary Classification Model.

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

To experiment different runs to evaluate metrics, you can alter the arguments, for example, 
expand the size of network by providing more `output` to the hidden layers. Or you 
may change the `drop_rate` or `train_batch_size` — all will alter the loss and 
accuracy of the network model.

To run the current program with just python and yet log all metrics, use
the following command:

`python keras/keras_nn_model.py`

`python keras/keras_nn_model.py --output=128 --epochs=10`

`python keras/keras_dnn/main_nn.py --output=128 --epochs=10`

It will log metrics and parameters in the `mlruns` directory. 

Alternatively, you can run using the `mlflow` command.

`mflow run . e -keras-nn-model`

`mlflow run . -e keras-nn-model -P drop_rate=0.3 -P output=128`

The next two examples are from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).
While the Jupyter notebooks can be found [here](https://github.com/fchollet/deep-learning-with-python-notebooks), I have modified the code 
to tailor for use with MLflow. The description and experimentation remain the same, hence it fits well with using MLflow to experiment
various capacity of networks layers and suggested parameters to evaluate the model.

## 2. Classifying Movie Reviews: a Keras binary classification example.

This part comprises of code samples found in Chapter 3, Section 5 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 
The borrowed code from the book has been modularized and adjusted to work with MLflow, and it fits well since Francois
suggests some experimentation parameters to tweak to see how the model metrics change.

Two-class classification, or binary classification, may be the most widely applied kind of machine learning problem. In this example, we 
will learn to classify movie reviews into "positive" reviews and "negative" reviews, just based on the text content of the reviews.

This example creates two types of models for you to work with. First, it creates a baseline model with default
parameters:

 * loss function as `rmstrop`
 * `binary_crossentropy` for metrics
 * learning rate as 0.001
 * a Keras neural network model with
    * An input layer with  input_shape (10000, )
    * 1 hidden layer with output = 32 
    * 1 Output layer with output = 1
    * All layers use `relu` as an activation function except for `sigmod` that is used in the final output layer.
  * epochs = 20; batch_size=512
 
 And the second model can me created for experimenting by changing any of the these parameters to measure the metrics:
 
  * Use 2 or more hidden layers
  * Use 4, 8, 12 or 16 epochs
  * Try hidden layers with output 32, 64 or 128 and see if that affects the metrics
  * Try to use the `mse` loss function instead of `binar_crossentropy`.
  
  In both cases, the model will create images for training and validation loss/accuracy images in the images directory
  
  ![Baseline Loss](./keras/images/baseline_loss.png)
  
  ![Baseline Accuracy](./keras/images/baseline_accuracy.png)
  
 ### How to Use MLflow to Experiment, Log Metrics and Artifacts
 
 To run the current program with just python and default or supplied parameter and yet log all metrics, use
the following command:

`cd imdbclassifier`

`python main_nn.py`

To experiment different runs, with different parameters suggested above, and evaluate new metrics, you can alter the arguments. For example, 
expand the size of network by providing more `output` to the hidden layers. Or you 
may change the `hidden_layers` or `epochs` or `loss` function — all will alter the loss and 
accuracy of the network model. For example,

`python main_nn.py --hidden_laysers=3 --output=32 --epochs=30 --loss=mse`

It will log metrics and parameters in the `mlruns` directory. 

Alternatively, you can run using the `mlflow` command.

**Note**: `mlflow run..` may take longer as it needs to create and setup an environment by and download and
installing dependency packages listed in `conda.yml`

`mlflow run keras/imdbclassifier -e main`

`mlflow run keras/imdbclassifier -e main -P hidden_layers=3 -P output=32 -P epochs=30 -P loss=mse`

 To view the output of either runs, launch the mlflow ui:
 
 `mlflow ui`
 
 These runs will not only log metrics for loss and accuracy but also log graphs generated from `matplotlib` for 
 perusal as part of visual artifacts.
 
 ![Saved Arficats](./keras/imdbclassifier/images/mlflow_ui_artifacts.png)
 
 Finally, you can run this in a Jupyter Notebook: 
 ![Jupyter Notebook](./keras/imdbclassifier/keras_binary_nn.ipynb)
 
 ### How Visualize TensorFlow Graphs with Tensorboard

If you have TensorBoard installed, you can also visualize the TensorFlow session graph created by the `train_models()` within the `train_nn.py`.  For example, after executing the statement `python main_nn.py`, you will see something similar to the following output:
```
Average Probability Results:
[0.30386349968910215, 0.88336000000000003]

Predictions Results:
[[ 0.35428655]
 [ 0.99231517]
 [ 0.86375767]
 ..., 
 [ 0.15689197]
 [ 0.24901576]
 [ 0.4418138 ]]
Writing TensorFlow events locally to /var/folders/0q/c_zjyddd4hn5j9jkv0jsjvl00000gp/T/tmp7af2qzw4

Uploading TensorFlow events as a run artifact.
loss function use binary_crossentropy
This model took 51.23427104949951  seconds to train and test.
```

You can extract the TensorBoard log directory with the line stating `Writing TensorFlow events locally to ...`.  That is, to run Tensorboard, you can run the following command:

```
tensorboard --logdir=/var/folders/0q/c_zjyddd4hn5j9jkv0jsjvl00000gp/T/tmp7af2qzw4
```

Click on **graph** and you can visualize and interact with your session graph.

![Tensorboard visualization](./images/visualize-graph-tensorboard.png)

 
 
## 3. Classifying Newswires: a multi-class Keras classification example

This contains the code samples found in Chapter 3, Section 5 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). 

----

In the above model we saw how to classify vector inputs into two mutually exclusive classes using a densely-connected neural network. 
But what happens when you have more than two classes? 

In this section, we will build a network to classify Reuters newswires into 46 different mutually-exclusive topics. Since we have many 
classes, this problem is an instance of "multi-class classification", and since each data point should be classified into only one 
category, the problem is more specifically an instance of "single-label, multi-class classification". If each data point could have 
belonged to multiple categories (in our case, topics) then we would be facing a "multi-label, multi-class classification" problem.


