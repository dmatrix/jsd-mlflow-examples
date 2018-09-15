#
# predict_nn.py
#    Reloads the model and predict sentiment based on
#    Based on main_nn.py
#

from imdbclassifier.parser_utils_nn import KParseArgs

import mlflow
import mlflow.keras
import numpy as np
from keras.datasets import imdb
import os

# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class KPredict():

    def __init__(self):
        return


    def predict_sentiment(self, args):
        # word_index is a dictionary mapping words to an integer index
        word_index = imdb.get_word_index()

        # flagVerbose
        flagVerbose = args.verbose

        # Extract review
        myReview = args.my_review

        # Take review and split into words list
        myReviewList = myReview.split()
        myReviewIdx = []

        # Enumerate through the word index
        for j, value in enumerate(myReviewList):
            value_index = word_index.get(value)
            # Only include words that exist in the imdb.word_index
            if str(value_index).isnumeric():
                # We encode the review; note that our indices were offset by 3
                # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
                value_index = value_index + 3
                if (flagVerbose == 1):
                    print("j: %s, word: %s, imdb_word_index: %s" % (j, value, value_index))

                # Create review index
                myReviewIdx.insert(j, value_index)

        # View my review index
        if (flagVerbose == 1):
            print("\n myReviewIdx: %s" % myReviewIdx)

        # OHE the sequence
        myVectorizedSeq = np.zeros((1, 10000))
        myVectorizedSeq[0, myReviewIdx] = 1.

        # View my vectorized sequence
        if (flagVerbose == 1):
            print("\n myVectorizedSeq: %s" % myVectorizedSeq)

        # Load model
        print("Loading Model...")
        path = args.load_model_path
        run_uuid=args.run_uuid
        model = mlflow.keras.load_model(path, run_id=run_uuid)

        # Print out predictions
        print("Predictions Results:")
        predictions = model.predict(myVectorizedSeq)
        print(predictions)


if __name__ == '__main__':
    #
    # Examples:
    #   Negative Sentiment:
    #       python predict_nn.py --load_model_path='/tmp/models' --my_review='this film was horrible, bad acting, even worse direction'
    #   Positive Sentiment:
    #       python predict_nn.py --load_model_path='/tmp/models' --my_review='this is a wonderful film with a great acting, beautiful cinematography, and amazing direction'
    #
    parser = KParseArgs()
    args = parser.parse_args()

    print("load model path:", args.load_model_path)
    print("run_uuid: ", args.run_uuid)
    print("my review:", args.my_review)
    print("verbose: ", args.verbose)

    KPredict().predict_sentiment(args)





