import utils_nn

import mlflow.sklearn
from time import time

from mlflow import log_metric

def compile_and_run_model(mdl, train_data, epochs=20, batch_size=128):
    """
    Compile the Keras NN model, fit and evaluate it
    :param mdl: A build Keras model
    :param train_data: train set for the model
    :param epochs: number of epochs. Default is 20
    :param batch_size: batch size for the training. Default is 128
    :return: list of prediction scores
    """

    mdl.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # train the model
    #
    mdl.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size, verbose=0)
    #
    # evaluate the network
    #
    score = mdl.evaluate(train_data[2], train_data[3], batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Predictions for Y:")
    print(mdl.predict(train_data[2][:5]))
    mdl.summary()

    return [score[0], score[1]]


def train(args):
    """
    Train the model and log all the metrics with mlflow
    :param args: command line arguments containing parameters for experimenting and tuning
    :return: results of prediction
    """
    input_dim = args.input_dim
    bs = args.bs
    output = args.output
    train_batch_size = args.train_batch_size
    drop_rate = args.drop_rate
    epochs = args.epochs

    data = utils_nn.gen_data(input_dim=input_dim, bsize=bs)
    model = utils_nn.build_model(in_dim=input_dim, drate=drop_rate, out=output)
    results = compile_and_run_model(model, data, epochs, train_batch_size)

    start_time = time()

    with mlflow.start_run():
        mlflow.log_param("drop_rate", drop_rate)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("size", bs)
        mlflow.log_param("output", output)
        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("loss", results[0])
        mlflow.log_param("acc", results[1])

    timed = time() - start_time

    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)

    return results

