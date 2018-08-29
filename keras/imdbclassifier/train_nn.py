from data_utils_nn import KIMDB_Data_Utils
from model_nn import KModel
from parser_utils_nn import KParseArgs
from graphs_nn import KPlot

import os
import sys
import mlflow
import mlflow.keras
import tensorflow as tf
import tempfile

from keras import optimizers
from keras import metrics

class KTrain():

    def __init__(self):
        return

    def compile_and_fit_model(self, model, x_train, y_train, epochs=20, batch_size=512, loss='binary_crossentropy',
                              optimizer='rmsprop', lr=0.0001, metrics=metrics.binary_accuracy,
                              verbose=1, save_model=0):
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

        if save_model:
            model_dir = self.get_directory_path("keras_models")
            self.keras_save_model(model, model_dir)
            print("Model is saved to %s" % model_dir)

        return history

    def keras_save_model(self, model, model_dir='/tmp'):
        """
        Convert Keras estimator to TensorFlow
        :type model_dir: object
        """

        mlflow.keras.save_model(model, model_dir)


    def evaulate_model(self,model, x_test, y_test):
        """
        Evaulate the model with unseen and untrained data
        :param model:
        :return: results of probability
        """

        return model.evaluate(x_test, y_test)

    def get_binary_loss(self, hist):
        loss = hist.history['loss']
        loss_val = loss[len(loss) - 1]
        return loss_val

    def get_binary_acc(self, hist):
        acc = hist.history['binary_accuracy']
        acc_value = acc[len(acc) - 1]

        return acc_value

    def get_validation_loss(self, hist):
        val_loss = hist.history['val_loss']
        val_loss_value = val_loss[len(val_loss) - 1]

        return val_loss_value

    def get_validation_acc(self, hist):
        val_acc = hist.history['val_binary_accuracy']
        val_acc_value = val_acc[len(val_acc) - 1]

        return val_acc_value


    def print_metrics(self, hist):

        acc_value = self.get_binary_acc(hist)
        loss_value = self.get_binary_loss(hist)

        val_acc_value = self.get_validation_acc(hist)

        val_loss_value = self.get_validation_loss(hist)

        print("Final metrics: binary_loss:%6.4f" % loss_value)
        print("Final metrics: binary_accuracy=%6.4f" % acc_value)
        print("Final metrics: validation_binary_loss:%6.4f" % val_loss_value)
        print("Final metrics: validation_binary_accuracy:%6.4f" % val_acc_value)

    def get_directory_path(self, dir_name):

        cwd = os.getcwd()
        dir = os.path.join(cwd, dir_name)

        return dir

    def train_models(self, args, base_line=True):
        """
        Train the model and log all the MLflow Metrics
        :param args: command line arguments. If no arguments then use default
        :param base_line: Default flag. Create Baseline model
        """
        # Create TensorFlow Session
        sess = tf.InteractiveSession()


        #
        # initialize some classes
        #
        kdata_cls = KIMDB_Data_Utils()
        ktrain_cls = KTrain()
        kplot_cls = KPlot()

        #
        # get IMDB Data
        #
        (train_data, train_labels), (test_data, test_labels) = kdata_cls.fetch_imdb_data()

        #
        # prepare and vectorize data
        #
        x_train = kdata_cls.prepare_vectorized_sequences(train_data)
        x_test = kdata_cls.prepare_vectorized_sequences(test_data)

        y_train = kdata_cls.prepare_vectorized_labels(train_labels)
        y_test = kdata_cls.prepare_vectorized_labels(test_labels)

        image_dir = ktrain_cls.get_directory_path("images")
        model_dir = ktrain_cls.get_directory_path("models")

        graph_label_loss = 'Baseline Model: Training and Validation Loss'
        graph_label_acc = 'Baseline Model: Training and Validation Accuracy'
        graph_image_loss_png = os.path.join(image_dir,'baseline_loss.png')
        graph_image_acc_png = os.path.join(image_dir, 'baseline_accuracy.png')

        if not base_line:
            graph_label_loss = 'Experimental: Training and Validation Loss'
            graph_label_acc = 'Experimental Model: Training and Validation Accuracy'
            graph_image_loss_png = os.path.join(image_dir, 'experimental_loss.png')
            graph_image_acc_png = os.path.join(image_dir,'experimental_accuracy.png')

        kmodel = KModel()
        if base_line:
            print("Baseline Model:")
            model = kmodel.build_basic_model()
        else:
            print("Experiment Model:")
            model = kmodel.build_experimental_model(args.hidden_layers, args.output)

        history = ktrain_cls.compile_and_fit_model(model, x_train, y_train, epochs=args.epochs, loss=args.loss)
        model.summary()
        ktrain_cls.print_metrics(history)

        figure_loss = kplot_cls.plot_loss_graph(history, graph_label_loss)
        figure_loss.savefig(graph_image_loss_png )

        figure_acc = kplot_cls.plot_accuracy_graph(history, graph_label_acc)
        figure_acc.savefig(graph_image_acc_png)

        results = ktrain_cls.evaulate_model(model, x_test, y_test)

        print("Average Probability Results:")
        print(results)

        print()
        print("Predictions Results:")
        predictions = model.predict(x_test)
        print(predictions)

        with mlflow.start_run():
            # log parameters
            mlflow.log_param("hidden_layers", args.hidden_layers)
            mlflow.log_param("output", args.output)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("loss_function", args.loss)
            # log metrics
            mlflow.log_metric("binary_loss", ktrain_cls.get_binary_loss(history))
            mlflow.log_metric("binary_acc",  ktrain_cls.get_binary_acc(history))
            mlflow.log_metric("validation_loss", ktrain_cls.get_binary_loss(history))
            mlflow.log_metric("validation_acc", ktrain_cls.get_validation_acc(history))
            mlflow.log_metric("average_loss", results[0])
            mlflow.log_metric("average_acc", results[1])
            # log artifacts
            mlflow.log_artifacts(image_dir, "images")
            # log model
            mlflow.keras.log_model(model, "models")
            # Write out tensorflow graph
            output_dir = tempfile.mkdtemp()
            print("Writing TensorFlow events locally to %s\n" % output_dir)
            writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
            print("Uploading TensorFlow events as a run artifact.")
            mlflow.log_artifacts(output_dir, artifact_path="events")

        print("loss function use", args.loss)

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    if flag:
        print("Using Default Baseline parameters")
    else:
        print("Using Experimental parameters")

    print("hidden_layers:", args.hidden_layers)
    print("output:", args.output)
    print("epochs:", args.epochs)
    print("loss:", args.loss)

    train_models_cls = KTrain().train_models(args, flag)





