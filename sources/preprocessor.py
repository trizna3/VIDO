import nltk
from nltk.corpus import stopwords

from sources.utils import *


def load_wordnet():
    nltk.download("wordnet")


def load_stopwords():
    nltk.download("stopwords")


class Preprocessor:
    """
    Data preprocessor.
    """

    def __init__(self, remove_stop_words=False, lowercase=False, lemmatize=False, stem=False):
        self.remove_stop_words = remove_stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.stem = stem

        load_wordnet()
        load_stopwords()
        pd.set_option('mode.chained_assignment', None)

    def preprocess(self, dataframe):
        stp_words = stopwords.words("english")
        lemmatizer = nltk.WordNetLemmatizer()
        stemmer = nltk.stem.PorterStemmer()

        for i in range(len(dataframe[KEY_TEXT])):
            text = dataframe.loc[:, KEY_TEXT][i]
            new_text = ""
            first_word = True
            for word in text.strip().split(" "):
                if self.remove_stop_words and word in stp_words:
                    continue
                if self.lowercase:
                    word = word.lower()
                if self.lemmatize:
                    word = lemmatizer.lemmatize(word)
                if self.stem:
                    word = stemmer.stem(word)

                if not first_word:
                    new_text += " "
                else:
                    first_word = False
                new_text += word

            dataframe.loc[:, KEY_TEXT][i] = new_text

        return dataframe
