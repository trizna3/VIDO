import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sources.ngram import *
from sources.preprocessor import *
from sacred import Experiment

ex = Experiment('vido_experiment', ingredients=[vido_ingredient])
NAME_NGRAM = "ngram"

# get data
train_fpath = get_train_path()
test_fpath = get_test_path()

# initialize model

@ex.automain
def main():
    preprocessor = Preprocessor()
    model = Ngram(NAME_NGRAM, preprocessor)
    # training
    model.train(train_fpath)

    # evaluation
    precision = model.test(test_fpath)
    return "Precision = {}".format(precision)
