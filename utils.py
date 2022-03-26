import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from itertools import chain
np.random.seed(42)


def frequency_filter(df, min_freq=1, lowercase=True):
    if lowercase:
        df['sentence'] = df['sentence'].str.lower()
        v = df['sentence'].str.split().tolist()
    else:
        v = df['sentence'].str.split().tolist()

    c = Counter(chain.from_iterable(v))
    df['sentence'] = [' '.join([j for j in i if c[j] > min_freq]) for i in v]

    return df


def stratified_split(X, y, train_size, test_size):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=12)

    for train, test in splitter.split(X, y):
        X_train_ss = X[train]
        y_train_ss = y[train]
        X_test_ss = X[test]
        y_test_ss = y[test]

    return X_train_ss, y_train_ss, X_test_ss, y_test_ss


def clean_url(tweet):
    """
    Clean tweets from URLs.
    """
    tweet = re.sub('<|endoftext|>', '.', tweet)
    tweet = re.sub('endoftext', '.', tweet)
    tweet = re.sub('|>', '', tweet)
    tweet = re.sub('<|', '.', tweet)
    tweet = tweet.split()
    clean_tweet = [x for x in tweet if not x.startswith("http")]
    return ' '.join(clean_tweet)


def CharLevel_Dataset(file=None, maxlen=200, lowercase=True, min_freq=1):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_to_index = {c: i + 1
                     for i, c in enumerate(alphabet)}

    trainset_df = pd.read_csv(file)  # , encoding="ISO-8859-1"
    trainset_df['sentence'] = trainset_df['sentence'].apply(clean_url)

    trainset_df = frequency_filter(df=trainset_df, min_freq=min_freq, lowercase=lowercase)
    sentences = list(trainset_df['sentence'])

    X = [[0] * maxlen for _ in range(len(sentences))]

    for sent_index, s in enumerate(sentences):
        for char_index, char in enumerate(s[:maxlen]):

            if char in alphabet:
                X[sent_index][char_index] = char_to_index[char]
            else:
                continue
    y = np.array(list(trainset_df['label']))
    return np.array(X), y
