from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sources.model import *


class Ngram(Model):
    """
    Ngram baseline model.
    """

    def __init__(self, name, preprocessor):
        super().__init__(name, preprocessor)
        self.name = name
        self.pipeline = None

    def fit(self, dataframe):
        """
        Trains on given labeled data.
        """
        self.pipeline = Pipeline([
            ('ngrams', TfidfVectorizer(ngram_range=(1, 1))),
            ('clf', SVC(C=1, gamma=0.75, kernel='rbf', random_state=0))
        ])
        self.pipeline.fit(dataframe[KEY_TEXT], dataframe[KEY_CHECK_WORTHINESS])

    def run(self, dataframe):
        """
        Model is fed inputs, writing outputs in the result file.
        """
        results_fpath = self.get_result_path()
        with open(results_fpath, "w") as results_file:
            predicted_distance = self.pipeline.decision_function(dataframe[KEY_TEXT])
            for i, line in dataframe.iterrows():
                dist = predicted_distance[i]
                results_file.write("{}\t{}\t{}\t{}\n".format(line[KEY_TOPIC_ID], line[KEY_TWEET_ID], dist, "ngram"))
