import train_nn
import argparse
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_layers", help="Number of Hidden Layers", action='store',  nargs='?', default=1, type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store',  nargs='?', default=16, type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20, type=int)
    parser.add_argument("--loss", help="Number of epochs for training", nargs='?', action='store', default='binary_crossentropy', type=str)

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

    train_nn.train_models(args, flag)
