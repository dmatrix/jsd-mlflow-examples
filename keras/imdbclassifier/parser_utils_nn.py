import argparse

class KParseArgs():

    def __init__(self):
        self.args = parser = argparse.ArgumentParser()

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--hidden_layers", help="Number of Hidden Layers", action='store', nargs='?', default=1,
                            type=int)
        self.parser.add_argument("--output", help="Output from First & Hidden Layers", action='store', nargs='?', default=16,
                            type=int)
        self.parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20,
                            type=int)
        self.parser.add_argument("--loss", help="Loss Function for the Gradients", nargs='?', action='store',
                            default='binary_crossentropy', type=str)
        self.parser.add_argument("--load_model_path", help="Load model path", nargs='?', action='store', default='/tmp', type=str)
        self.parser.add_argument("--my_review", help="Type in your review", nargs='?', action='store', default='this film was horrible, bad acting, even worse direction', type=str)
        self.parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)
        self.parser.add_argument("--run_uuid", help="Specify the MLflow Run ID", nargs='?', action='store', default=None, type=str)
        self.parser.add_argument("--tracking_server", help="Specify the MLflow Tracking Server", nargs='?', action='store', default=None, type=str)

    def parse_args(self):
        return self.parser.parse_args()

    def parse_args_list(self, args_list):
        return self.parser.parse_args(args_list)

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    print ("====default args====:")
    print(args)
    print ("----empty list (default args)-----")
    args = parser.parse_args_list([])
    print(args)
    print("+++++default + list of args+++++")
    args = parser.parse_args_list(['--hidden_layers', '3', '--epochs', '10', '--output', '32', '--loss', 'mse', '--tracking_server', '123.45.6.8'])
    print(args)