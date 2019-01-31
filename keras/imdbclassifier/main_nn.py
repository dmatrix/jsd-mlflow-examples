from imdbclassifier.train_nn import KTrain
from imdbclassifier.parser_utils_nn import KParseArgs
from time import time
import sys
import os

# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    parser = KParseArgs()
    args = parser.parse_args()

    start_time = time()

    flag = len(sys.argv) == 1

    if flag:
        print("Using Default Baseline parameters")
    else:
        print("Using Experimental parameters")

    print("hidden_layers:", args.hidden_layers)
    print("output:", args.output)
    print("epochs:", args.epochs)
    print("loss:", args.loss)
    print("experiment_name:", args.experiment_name)

    train_models_cls = KTrain().train_models(args, flag)

    timed = time() - start_time

    print("This model took", timed, " seconds to train and test.")