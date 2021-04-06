import nltk
from nltk.tokenize import TweetTokenizer
from gensim.corpora import Dictionary
import itertools
from collections import defaultdict
import pandas as pd

t = TweetTokenizer()

def tokenize(tweet):
    tokens = t.tokenize(tweet)
    return [word for word in tokens]

def tokenize_n_grams(string, n_gram):
    tokens = ngrams(string, n_gram)
    return [word for word in tokens]

def get_token_frequency(series):
    corpus_lists = [doc for doc in series.dropna() if doc]
    dictionary = Dictionary(corpus_lists)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_lists]
    token_freq_bow = defaultdict(int)

    for token_id, token_sum in itertools.chain.from_iterable(corpus_bow):
        token_freq_bow[token_id] += token_sum

    # Create dataframes
    df = pd.DataFrame(list(token_freq_bow.items()), columns = ['token_id', 'token_count']).assign(
        token = lambda df1: df1.apply(lambda df2: dictionary.get(df2.token_id), axis = 1),
        doc_appeared = lambda df1: df1.apply(lambda df2: dictionary.dfs[df2.token_id], axis = 1)).reindex(
            labels = ['token_id', 'token', 'token_count', 'doc_appeared'], axis = 1).set_index('token_id')
    
    return df