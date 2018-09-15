#
# reproduce_run_nn.py
#    Fetches the model run_uuid from the tracking server, reads the parameters saved,
#    and recreates the experiments and runs. In essence it trains the model and predicts
#    with all the parameters used from previous experiments.
#

from imdbclassifier.train_nn import KTrain
from imdbclassifier.parser_utils_nn import KParseArgs
import mlflow.tracking

import sys

class KReproduce(KTrain):
    def __init__(self):
        super().__init__()
        return

    def train_models(self, args, base_line=True):
        super().train_models(args, base_line)

    def get_run_data(self, run_uuid, tracking_uri=None):
        client = mlflow.tracking.MlflowClient(tracking_uri)
        run_entries = client.get_run(run_uuid)
        run_data = run_entries.data
        return run_data

    def build_run_args_list(self, run_data):
        a_list = []
        for p in run_data.params:
            if p.key == 'loss_function':
                a_list.append('--loss')
            else:
                a_list.append('--' + p.key)
            a_list.append(p.value)
        return a_list

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    cls = KReproduce()
    data = cls.get_run_data(args.run_uuid, args.tracking_server)
    args_list = cls.build_run_args_list(data)

    print("run_uuid:", args.run_uuid)

    args = parser.parse_args_list(args_list)

    print("hidden_layers:", args.hidden_layers)
    print("output:", args.output)
    print("epochs:", args.epochs)
    print("loss:", args.loss)
    print("load model path:", args.load_model_path)
    print("tracking server:", args.tracking_server)

    KReproduce().train_models(args, flag)

