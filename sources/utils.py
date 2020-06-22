import pandas as pd

TRAIN_PATH_v1 = "../data/v1/training.tsv"
TEST_PATH_v1 = "../data/v1/dev.tsv"
TRAIN_PATH_v2 = "../data/v2/training.tsv"
TEST_PATH_v2 = "../data/v2/dev.tsv"

KEY_TEXT = "tweet_text"
KEY_TWEET_ID = "tweet_id"
KEY_TOPIC_ID = "topic_id"
KEY_CHECK_WORTHINESS = "check_worthiness"


def get_train_path(version="v1"):
    if version == "v1":
        return TRAIN_PATH_v1
    elif version == "v2":
        return TRAIN_PATH_v2
    else:
        raise Exception("invalid version")


def get_test_path(version="v1"):
    if version == "v1":
        return TEST_PATH_v1
    elif version == "v2":
        return TEST_PATH_v2
    else:
        raise Exception("invalid version")


def read_datafile(fpath):
    """
    Reads given file, returns pandas.dataframe object.
    """
    return pd.read_csv(fpath, sep='\t')
