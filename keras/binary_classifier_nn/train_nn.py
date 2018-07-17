import data_utils_nn
import model_nn
import graphs_nn
import matplotlib
import os
import sys
import mlflow

from time import time

from mlflow import log_metric

import argparse

from keras import optimizers
from keras import losses
from keras import metrics


def compile_and_fit_model(model, x_train, y_train, epochs=20, batch_size=512, loss=losses.binary_crossentropy,
                          optimizer='rmsprop', lr=0.0001, metrics=metrics.binary_accuracy,
                          verbose=1):
    #
    # generate validation data and training data
    #
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partail_y_train = y_train[10000:]

    if optimizer == 'rmsprop':
        opt = optimizers.RMSprop(lr=lr)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[metrics])

    #
    # fit the model: use part of the training data and use validation for unseen data
    #
    history = model.fit(partial_x_train,
                        partail_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        validation_data=(x_val, y_val))

    return history

def evaulate_model(model, x_test, y_test):
    """
    Evaulate the model with unseen and untrained data
    :param model:
    :return: results of probability
    """

    return model.evaluate(x_test, y_test)

def get_binary_loss(hist):
    loss = hist.history['loss']
    loss_val = loss[len(loss) - 1]
    return loss_val

def get_binary_acc(hist):
    acc = hist.history['binary_accuracy']
    acc_value = acc[len(acc) - 1]

    return acc_value

def get_validation_loss(hist):
    val_loss = hist.history['val_loss']
    val_loss_value = val_loss[len(val_loss) - 1]

    return val_loss_value

def get_validation_acc(hist):
    val_acc = hist.history['val_binary_accuracy']
    val_acc_value = val_acc[len(val_acc) - 1]

    return val_acc_value


def print_metrics(hist):

    acc_value = get_binary_acc(hist)
    loss_value = get_binary_loss(hist)

    val_acc_value = get_validation_acc(hist)

    val_loss_value = get_validation_loss(hist)

    print("Final metrics: binary_loss:%6.4f" % loss_value)
    print("Final metrics: binary_accuracy=%6.4f" % acc_value)
    print("Final metrics: validation_binary_loss:%6.4f" % val_loss_value)
    print("Final metrics: validation_binary_accuracy:%6.4f" % val_acc_value)

def get_images_directory_path():

    cwd = os.getcwd()

    image_dir = os.path.join(cwd, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir, mode=0o755)

    return image_dir

def train_models(args, base_line=True):
    """
    Train the model and log all the MLflow Metrics
    :param args: command line arguments. If no arguments then use default
    :param base_line: Default flag. Create Baseline model
    """

    start_time = time()

    (train_data, train_labels), (test_data, test_labels) = data_utils_nn.fetch_imdb_data()

    x_train = data_utils_nn.prepare_vectorized_sequences(train_data)
    x_test = data_utils_nn.prepare_vectorized_sequences(test_data)

    y_train = data_utils_nn.prepare_vectorized_labels(train_labels)
    y_test = data_utils_nn.prepare_vectorized_labels(test_labels)

    image_dir = get_images_directory_path()

    graph_label_loss = 'Baseline Model: Training and Validation Loss'
    graph_label_acc = 'Baseline Model: Training and Validation Accuracy'
    graph_image_loss_png = os.path.join(image_dir,'baseline_loss.png')
    graph_image_acc_png = os.path.join(image_dir, 'baseline_accuracy.png')

    if not base_line:
        graph_label_loss = 'Experimental: Training and Validation Loss'
        graph_label_acc = 'Experimental Model: Training and Validation Accuracy'
        graph_image_loss_png = os.path.join(image_dir, 'experimental_loss.png')
        graph_image_acc_png = os.path.join(image_dir,'experimental_accuracy.png')

    if base_line:
        print("Baseline Model:")
        model = model_nn.build_basic_model()
    else:
        print("Experiment Model:")
        model = model_nn.build_experimental_model(args.hidden_layers, args.output)

    history = compile_and_fit_model(model, x_train, y_train, epochs=args.epochs, loss=losses.binary_crossentropy)
    model.summary()
    print_metrics(history)

    figure_loss = graphs_nn.plot_loss_graph(history, graph_label_loss)
    figure_loss.savefig(graph_image_loss_png )

    figure_acc = graphs_nn.plot_accuracy_graph(history, graph_label_acc)
    figure_acc.savefig(graph_image_acc_png )

    results = evaulate_model(model, x_test, y_test)

    print("Average Probability Results:")
    print(results)

    print()
    print("Predictions Results:")
    predictions = model.predict(x_test)
    print(predictions)

    timed = time() - start_time

    with mlflow.start_run():
        mlflow.log_param("hidden_layers", args.hidden_layers)
        mlflow.log_param("output", args.output)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("loss function", args.loss)
        mlflow.log_param("binary_loss", get_binary_loss(history))
        mlflow.log_param("binary_acc",  get_binary_acc(history))
        mlflow.log_param("validation_loss", get_binary_loss(history))
        mlflow.log_param("validation_acc", get_validation_acc(history))
        mlflow.log_param("results", results)
        mlflow.log_artifacts(image_dir)
        mlflow.log_metric("Time to run", timed)

    print("This model took", timed, "seconds to train and test.")

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    flag = len(sys.argv) == 0
    train_models(args, flag)



