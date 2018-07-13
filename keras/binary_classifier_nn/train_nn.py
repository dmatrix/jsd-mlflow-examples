import data_utils_nn
import model_nn
import graphs_nn
import matplotlib

from keras import optimizers
from keras import losses
from keras import metrics


def compile_and_fit_model(model, x_train, y_train, epochs=20, batch_size=512, loss=losses.binary_crossentropy,
                          optimizer='rmsprop', lr=0.0001, metrics=metrics.binary_accuracy,
                          verbose=0):
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

def print_metrics(hist):
    acc = hist.history['binary_accuracy']
    acc_value = acc[len(acc) - 1]

    loss = hist.history['loss']
    loss_value = loss[len(loss) - 1]

    val_acc = hist.history['val_binary_accuracy']
    val_acc_value = val_acc[len(val_acc) - 1]

    val_loss = hist.history['val_loss']
    val_loss_value = val_loss[len(val_loss) - 1]

    print("Final metrics: binary_loss:%6.4f" % loss_value)
    print("Final metrics: binary_accuracy=%6.4f" % acc_value)
    print("Final metrics: validation_binary_loss:%6.4f" % val_loss_value)
    print("Final metrics: validation_binary_accuracy:%6.4f" % val_acc_value)


if __name__ == '__main__':
    #
    # main used for testing the functions
    #

    (train_data, train_labels), (test_data, test_labels) = data_utils_nn.fetch_imdb_data()

    x_train = data_utils_nn.prepare_vectorized_sequences(train_data)
    x_test = data_utils_nn.prepare_vectorized_sequences(test_data)

    y_train = data_utils_nn.prepare_vectorized_labels(train_labels)
    y_test = data_utils_nn.prepare_vectorized_labels(test_labels)

    model = model_nn.build_basic_model()

    print("Base line Model:")
    model.summary()
    history = compile_and_fit_model(model, x_train, y_train)
    print_metrics(history)

    basic_figure_loss = graphs_nn.plot_loss_graph(history, "Baseline Model: Training and Validation loss")
    basic_figure_loss.savefig('images/baseline_loss.png')

    basic_figure_acc = graphs_nn.plot_accuracy_graph(history, "Baseline Model: Training and Validation accuracy")
    basic_figure_acc.savefig('images/baseline_accuracy.png')

    results = evaulate_model(model, x_test, y_test)

    print("Baseline Probability Results")
    print(results)


    print()

    print("Experiment Model:")
    custom_model = model_nn.build_experimental_model(3, 32)

    custom_history = compile_and_fit_model(custom_model, x_train, y_train, loss=losses.binary_crossentropy)
    custom_model.summary()
    print_metrics(custom_history)

    custom_figure_loss = graphs_nn.plot_loss_graph(custom_history, "Experimental Model: Training and Validation loss")
    custom_figure_loss.savefig('images/experiment_loss.png')

    custom_figure_acc = graphs_nn.plot_accuracy_graph(history, "Experiment Model: Training and Validation accuracy")
    custom_figure_acc.savefig('images/experiment_accuracy.png')

    results = evaulate_model(custom_model, x_test, y_test)

    print("Average Probability Results:")
    print(results)

    print()
    print("Predictions Results:")
    predictions = custom_model.predict(x_test)
    print(predictions)









