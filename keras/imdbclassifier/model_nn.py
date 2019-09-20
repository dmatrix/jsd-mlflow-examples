from keras import models
from keras import layers

class KModel():

    def __init__(self):
        return

    def build_basic_model(self):

        """
        Build the base line model with one input layer, one hidden layer, and one output layer, with
        16, 16, and 1 output neurons. Default activation functions for input and hidden layer are relu
        and sigmoid respectively
        :return: a Keras network model
        """

        base_model = models.Sequential()
        base_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        base_model.add(layers.Dense(16, activation='relu'))
        base_model.add(layers.Dense(1, activation='sigmoid'))

        return base_model

    def build_experimental_model(self, hidden_layers=1, output=16, activation='relu'):

        exp_model = models.Sequential()
        # add the input layers
        exp_model.add(layers.Dense(output, activation=activation, input_shape=(10000,)))
        # add hidden layers
        for i in range(0, hidden_layers):
            exp_model.add(layers.Dense(output, activation=activation))
        # add output layer
        exp_model.add(layers.Dense(1, activation='sigmoid'))

        return exp_model

if __name__ == '__main__':

    mmaker = KModel()
    # build the basic model
    model = mmaker.build_basic_model()
    model.summary()
    # build an experimental
    custom_model = mmaker.build_experimental_model(3, 32, 'tanh')
    custom_model.summary()
