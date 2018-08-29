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
        self.parser.add_argument("--loss", help="Number of epochs for training", nargs='?', action='store',
                            default='binary_crossentropy', type=str)
        self.parser.add_argument("--load_model_path", help="Load model path", nargs='?', action='store', default='/tmp', type=str)
        self.parser.add_argument("--my_review", help="Type in your review", nargs='?', action='store', default='this film was horrible, bad acting, even worse direction', type=str)
        self.parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=False, type=bool)

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

    print(args)

    args = parser.parse_args_list([])
    print(args)

    args = parser.parse_args_list(['--hidden_layers', '3', '--epochs', '10', '--output', '32', '--loss', 'mse'])
    args = parser.parse_args()
    print(args)