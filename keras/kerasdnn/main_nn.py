import train_nn
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_rate", help="Drop rate", nargs='?', action='store', default=0.5, type=float)
    parser.add_argument("--input_dim", help="Input dimension for the network.", action='store', nargs='?', default=20, type=int)
    parser.add_argument("--bs", help="Number of rows or size of the tensor", action='store', nargs='?', default=1000, type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store',  nargs='?', default=64, type=int)
    parser.add_argument("--train_batch_size", help="Training Batch Size", nargs='?', action='store', default=128, type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20, type=int)

    args = parser.parse_args()
    print("drop_rate", args.drop_rate)
    print("input_dim", args.input_dim)
    print("size", args.bs)
    print("output", args.output)
    print("train_batch_size", args.train_batch_size)
    print("epochs", args.epochs)

    res = train_nn.train(args)

    print(res)

