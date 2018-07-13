from keras import models
from keras import layers

def build_basic_model():

    """
    Build the base line model with one input layer, one hidden layer, and one output layer, with
    16, 16, and 1 output neurons. Default activation functions for input and hidden layer are relu
    and sigmoid respectively
    :return: a Keras network model
    """

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def build_experimental_model(hidden_layers=2, output=16, activation='relu'):

    model = models.Sequential()
    #add the input layers
    model.add(layers.Dense(output, activation='relu', input_shape=(10000,)))
    # add hidden layers
    for i in range(0, hidden_layers):
        model.add(layers.Dense(output, activation=activation))
    #add output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':

    model = build_basic_model()
    model.summary()

    custom_model = build_experimental_model(3, 32, 'tanh')
    custom_model.summary()