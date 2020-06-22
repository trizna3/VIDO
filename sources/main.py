from sources.ngram import *
from sources.preprocessor import *

NAME_NGRAM = "ngram"

# get data
train_fpath = get_train_path()
test_fpath = get_test_path()

# initialize model
preprocessor = Preprocessor(
    remove_stop_words=False,
    lowercase=True,
    lemmatize=True,
    stem=False
)
model = Ngram(NAME_NGRAM, preprocessor)

# training
model.train(train_fpath)

# evaluation
precision = model.test(test_fpath)
print("precision = {}".format(precision))