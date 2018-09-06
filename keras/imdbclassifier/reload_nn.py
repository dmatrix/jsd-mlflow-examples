#
# reload_nn.py
#    Reloads the model you had created and executes it
#    Based on main_nn.py
#

from train_nn import KTrain
from parser_utils_nn import KParseArgs

import mlflow.keras
import sys

class KReload(KTrain):
    def __init__(self):
        super().__init__()
        return

    def train_models(self, args, base_line=True):
        super().train_models(args, base_line)

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    print("hidden_layers:", args.hidden_layers)
    print("output:", args.output)
    print("epochs:", args.epochs)
    print("load model path:", args.load_model_path)
    print("tracking server:", args.tracking_server)

    KReload().train_models(args, flag)




