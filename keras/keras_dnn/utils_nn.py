import numpy as np

from keras import Sequential
from keras.layers import Dense, Dropout

def gen_data(input_dim=20, bsize=1000):
    """
    Generates random fake data for train X, Y and testing or validation X, Y
    :param input_dim: Input dimension for the nueral network. Default is 20
    :param bsize: Size or number of rows in the data. Default is 1000
    :return: list [x_train, y_train, x_test, y_test]
    """

    # Generate dummy data for training and test set
    x_train = np.random.random((bsize, input_dim))
    y_train = np.random.randint(2, size=(bsize, 1))
    x_test = np.random.random((int(bsize * 0.10), input_dim))
    y_test = np.random.randint(2, size=(int(bsize * 0.10), 1))

    return [x_train, y_train, x_test, y_test]


def build_model(in_dim=20, drate=0.5, out=64):
    """
    Build the Keras Dense Layer Network.
    :param in_dim: Input dimension of the first layer. Default is 20
    :param drate: dropout rate for the neural network. Default is 0.5
    :param out: Output dimension of each layer in the network. Default is 64
    :return: returns a fully connected dense layer with specified dropout layers
    """
    mdl = Sequential()
    mdl.add(Dense(out, input_dim=in_dim, activation='relu'))
    if drate:
        mdl.add(Dropout(drate))
    mdl.add(Dense(out, activation='relu'))
    if drate:
        mdl.add(Dropout(drate))
    mdl.add(Dense(1, activation='sigmoid'))

    return mdl

