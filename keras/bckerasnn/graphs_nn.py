import matplotlib.pyplot as plt

class KPlot():
    
    def __init__(self):
        return

    def plot_loss_graph(self,history, title):
        """
        Generate a matplotlib graph for the loss and accuracy metrics
        :param histroy:
        :return: instance of a graph
        """

        acc = history.history['binary_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, loss, 'bo')

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        return fig

    def plot_accuracy_graph(elf, history, title):

        plt.clf()
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']

        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, acc, 'bo')

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        return fig








