import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sources.ngram import *
from sources.preprocessor import *
from sacred import Experiment
from sources.fasttext import *

ex = Experiment('vido_experiment', ingredients=[vido_ingredient])
NAME_NGRAM = "ngram"
NAME_FASTEXT = "fastext"

# get data
train_fpath = get_train_path()
test_fpath = get_test_path()

# initialize model

TYPE = "FASTTEXT"
# TYPE = "NGRAM"


@ex.automain
def main():
    preprocessor = Preprocessor()

    if TYPE == "NGRAM":
        model = Ngram(NAME_NGRAM, preprocessor)
    # training
    else:
        model = FastText(NAME_FASTEXT, preprocessor)

    model.train(train_fpath)
    precision = model.test(test_fpath)
    return "Precision = {}".format(precision)
