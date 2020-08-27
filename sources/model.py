from sources.utils import *
from sources.scorer.main import evaluate
from sources.format_checker.main import check_format

DEFAULT_RESULT_PATH = "results/result_{}.tsv"


class Model:
    """
    Abstract class for nlp models.
    """

    def __init__(self, name, preprocessor):
        self.name = name
        self.preprocessor = preprocessor

    def fit(self, data):
        raise NotImplementedError

    def run(self, data):
        raise NotImplementedError

    def train(self, train_data_path):
        train_data = read_datafile(train_data_path)
        train_data = self.preprocessor.preprocess(train_data)
        self.fit(train_data)

    def test(self, test_data_path):
        """
        Feeds the model, writing outputs in the result file. Then measures the score of it's performance.
        """
        test_data = read_datafile(test_data_path)
        test_data = self.preprocessor.preprocess(test_data)
        self.run(test_data)
        if check_format(self.get_result_path()):
            thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(test_data_path,
                                                                                            self.get_result_path())
            return avg_precision

    def get_result_path(self):
        return DEFAULT_RESULT_PATH.format(self.name)
