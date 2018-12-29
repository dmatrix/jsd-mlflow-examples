import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import layers
import numpy as np

class Data():

    def __init__(self, ndim, input_units, output_units):
        self.data = np.random.random((ndim, input_units))
        self.labels = np.random.random((ndim, output_units))

        self.dataset = tf.data.Dataset.from_tensor_slices((self.data, self.labels))
        self.dataset = self.dataset.batch(input_units).repeat()

        self.val_data = np.random.random((100, input_units))
        self.val_labels = np.random.random((100, output_units))

        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.val_data, self.val_labels))
        self.val_dataset = self.val_dataset.batch(32).repeat()

    def get_training_dataset(self):
        return self.dataset

    def get_validation_dataset(self):
        return self.val_dataset

class Model():

    def __init__(self, input_units=32, output_units=10, ndim=1000):
        self.input_units = input_units
        self.output_units = output_units
        self.ndim = ndim
        #create m densely connected model
        self.model = tf.keras.Sequential()
        # Add a densely-connected layer with input units to the model:
        self.model.add(layers.Dense(input_units, activation='relu'))
        # Add another layer
        self.model.add(layers.Dense(input_units, activation='relu'))
        # Add a softmax layer with 10 output units:
        self.model.add(layers.Dense(output_units, activation='softmax'))

    def compile(self, optimizer, loss='mse', metrics=['mae']):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, ds, vds):
        self.model.fit(ds, epochs=10, steps_per_epoch=30, validation_data=vds, validation_steps=3)

    def print_versions(self):
        print()
        print(tf.VERSION)
        print(tf.keras.__version__)

if __name__ == '__main__':

    model = Model()
    parameters = dict(losses=['mse', 'binary_crossentropy', 'categorical_crossentropy'],
                           metrics=[['mae'], ['accuracy'], ['accuracy']],
                           optimizers=[tf.train.AdamOptimizer(0.01),
                                       tf.train.RMSPropOptimizer(0.01),
                                       tf.train.RMSPropOptimizer(0.01)])
    for i in range(len(parameters)):
        print('{}, {}, {}, {}'.format(i,
            parameters['optimizers'][i],
            parameters['losses'][i],
            parameters['metrics'][i]))
        model.compile(parameters['optimizers'][i],
                      parameters['losses'][i],
                      parameters['metrics'][i])
        # get Data
        data = Data(input_units=32, output_units=10, ndim=1000)
        dataset = data.get_training_dataset()
        val_dataset = data.get_validation_dataset()
        # fit the model with
        model.fit(dataset, val_dataset)
        print()
    model.print_versions()


