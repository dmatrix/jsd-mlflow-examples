from keras.datasets import imdb
import numpy as np

"""
Part of this module is derived and borrowed heavily from Francois Chollet's book Deep Learning Python. Original code
can be found at :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb

We'll be working with "IMDB dataset", a set of 50,000 highly-polarized reviews from the Internet Movie Database. They are split into 
25,000 reviews for training and 25,000 reviews for testing, each set consisting in 50% negative and 50% positive reviews.
Why do we have these two separate training and test sets? You should never test a machine learning model on the same data 
that you used to train it! Just because a model performs well on its training data doesn't mean that it will perform well 
on data it has never seen, and what you actually care about is your model's performance on new data (since you already 
know the labels of your training data -- obviously you don't need your model to predict those). For instance, it is 
possible that your model could end up merely memorizing a mapping between your training samples and their targets -- 
which would be completely useless for the task of predicting targets for data never seen before. We will go over this 
point in much more detail in the next chapter.

Just like the MNIST dataset, the IMDB dataset comes packaged with Keras. It has already been preprocessed: the reviews 
(sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary.

The following code will load the dataset (when you run it for the first time, about 80MB of data will be downloaded to your 
machine)
"""

class KIMDB_Data_Utils():

    def __init__(self):
        return

    def fetch_imdb_data(self, num_words=10000):
        """
        :param num_words: This arguments means that we want to keep the top 10,000 most frequently occuring words in the training data. Rare words will be discarded
        :return: The variables train_data and test_data are lists of reviews, each review being a list of word indices (encoding a sequence of words).  train_labels and test_labels are lists of 0s and 1s, where 0 stands for "negative" \
        and 1 stands for "positive":
        """
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

        return (train_data, train_labels), (test_data, test_labels)

    def decode_review(self, train_data, index=0):
        """
        Return a decoded review
        :param index: is index into mapping of words into the integer index
        :return: a string matching the review
        """
        # word_index is a dictionary mapping words to an integer index
        word_index = imdb.get_word_index()
        # We reverse it, mapping integer indices to words
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # We decode the review; note that our indices were offset by 3
        # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
        decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[index]])

        return decoded_review

    def prepare_vectorized_sequences(self, sequences, dimension=10000):
        """
        We cannot feed lists of integers into a neural network. We have to turn our lists into tensors. One way is to convert the sequence
        into tensors using Numpy. Also, we are going to use one-hot-encode our lists into vectors of 0s and 1s. That is, for instance turning the sequence
        [3, 5] into a 10,000-dimensional vector that would be all-zeros except for indices 3 and 5, which would be ones. Then we could use as first layer in our
        network a Dense layer, capable of handling floating point vector data.
        :param sequences: this is the sequence we want to convert
        :param dimension: size of the sequence
        :return: list of one-hot-encoded vector []
        """
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # set specific indices of results[i] to 1s

        return results

    def prepare_vectorized_labels(self, labels):
        """
        labels are scalars so we can just use numpy as arrays of type float
        :param labels: label data
        :return: numpy array
        """
        return np.asarray(labels).astype('float32')
#
# Test the functions
#
if __name__ == '__main__':
    # create a class handle
    kdata_cls = KIMDB_Data_Utils()
    (train_data, train_labels), (test_data, test_labels) = kdata_cls.fetch_imdb_data(num_words=10000)
    print(train_data[0])
    print(len(train_data))
    decoded = kdata_cls.decode_review(train_data)
    print(decoded)
    x_train = kdata_cls.prepare_vectorized_sequences(train_data)
    x_test = kdata_cls.prepare_vectorized_sequences(test_data)
    print(x_train[0])
    print(x_test[0])
    y_train = kdata_cls.prepare_vectorized_labels(train_labels)
    y_test = kdata_cls.prepare_vectorized_labels(test_labels)
    print(y_train)
    print(y_test)