from train_nn import KTrain
from parser_utils_nn import KParseArgs
import sys


if __name__ == '__main__':

    parser = KParseArgs()
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

    train_models_cls = KTrain().train_models(args, flag)
