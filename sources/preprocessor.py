import nltk
from nltk.corpus import stopwords
from sacred import Ingredient

from sources.utils import *

vido_ingredient = Ingredient('vido')

def load_wordnet():
    nltk.download("wordnet")


def load_stopwords():
    nltk.download("stopwords")

@vido_ingredient.config
def cfg():
    remove_stop_words = False
    lowercase = True
    lemmatize = True
    stem = False

@vido_ingredient.named_config
def full_preprocessing():
    remove_stop_words = True
    lowercase = True
    lemmatize = True
    stem = True

class Preprocessor:
    """
    Data preprocessor.
    """

    def __init__(self):
        load_wordnet()
        load_stopwords()
        pd.set_option('mode.chained_assignment', None)

    @vido_ingredient.capture
    def preprocess(self, dataframe, remove_stop_words, lowercase, lemmatize, stem):
        stp_words = stopwords.words("english")
        lemmatizer = nltk.WordNetLemmatizer()
        stemmer = nltk.stem.PorterStemmer()

        for i in range(len(dataframe[KEY_TEXT])):
            text = dataframe.loc[:, KEY_TEXT][i]
            new_text = ""
            first_word = True
            for word in text.strip().split(" "):
                if remove_stop_words and word in stp_words:
                    continue
                if lowercase:
                    word = word.lower()
                if lemmatize:
                    word = lemmatizer.lemmatize(word)
                if stem:
                    word = stemmer.stem(word)

                if not first_word:
                    new_text += " "
                else:
                    first_word = False
                new_text += word

            dataframe.loc[:, KEY_TEXT][i] = new_text

        return dataframe
