import os
import subprocess
from sources.model import *
from sources.utils import *


class FastText(Model):

    def __init__(self, name, preprocessor):
        super().__init__(name, preprocessor)
        self.name = name
        self.pipeline = None

    def fit(self, data):

        # process = subprocess.Popen(['../fastText/fasttext', 'supervised','-minCount 2', '-wordNgrams 3','-minn 3', '-maxn 8', '-lr 0.7', '-dim 100', '-epoch 25', f'-input ./{self.name}.train', f'-output ./{self.name}.model',], 
        #    stdout=subprocess.PIPE,
        #    universal_newlines=True)

        stream = os.popen(
            f'../fastText/fasttext supervised -minCount 2 -wordNgrams 3 -minn 3 -maxn 8 -lr 0.7 -dim 100 -epoch 25 f-input ./{self.name}.train f-output ./{self.name}.model')

        output = stream.read()
        print(output)

        # while True:
        #     output = process.stdout.readline()
        #     print(output.strip())
        #     # Do something else
        #     return_code = process.poll()
        #     if return_code is not None:
        #         print('RETURN CODE', return_code)
        #         # Process has finished, read rest of the output 
        #         for output in process.stdout.readlines():
        #             print(output.strip())
        #         break

    def run(self, data):
        # process = subprocess.Popen(['../fastText/fasttext', f'test ./{self.name}.model', f'./{self.name}.valid','3'], 
        #                    stdout=subprocess.PIPE,
        #                    universal_newlines=True)

        # while True:
        #     output = process.stdout.readline()
        #     print(output.strip())
        #     # Do something else
        #     return_code = process.poll()
        #     if return_code is not None:
        #         print('RETURN CODE', return_code)
        #         # Process has finished, read rest of the output 
        #         for output in process.stdout.readlines():
        #             print(output.strip())
        #         break

        stream = os.popen(f'../fastText/fasttext test ./{self.name}.model ./{self.name}.valid 3')
        output = stream.read()
        print(output)

    def train(self, train_data_path):
        train_data = read_datafile(train_data_path)
        train_data = self.preprocessor.preprocess(train_data)

        data = '__label__' + train_data['claim'].astype(str) + train_data['check_worthiness'].astype(str) + ' ' + \
               train_data['tweet_text']

        with open(f'{self.name}.train', 'w') as f:
            for x in data:
                self.print_sequence(x, f)

        self.fit(train_data)

    def test(self, test_data_path):
        """
        Feeds the model, writing outputs in the result file. Then measures the score of it's performance.
        """
        test_data = read_datafile(test_data_path)
        test_data = self.preprocessor.preprocess(test_data)

        data = '__label__' + test_data['claim'].astype(str) + test_data['check_worthiness'].astype(str) + ' ' + \
               test_data['tweet_text']

        with open(f'{self.name}.valid', 'w') as f:
            for x in data:
                self.print_sequence(x, f)
        self.run(data)

    def get_result_path(self):
        return DEFAULT_RESULT_PATH.format(self.name)

    def print_sequence(self, seq, f):
        for char in seq:
            try:
                print(char, file=f, end='', )
            except UnicodeEncodeError:  # skip character
                # print("skipped character = {}".format(char))
                pass
        print('', file=f)
