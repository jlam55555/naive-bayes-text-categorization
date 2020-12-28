from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

# from training labels, get list of all classes
# each class will store its own vocabulary
classes = {}

# parse training labels
with open(input('Enter training file: '), 'r') as training_file:
    for line in training_file:
        file, category = line.rstrip('\n').split(' ')

        # just get filename, ignore rest of path
        file = file.split('/')[-1]

        # TODO: add training labels, classes to some dict

# Equations for Naive Bayes:
#   P(C) = N_C / N_doc
#   P(w_i|C) = (count(w_i, C) + 1) / (\sum_{w\in V}{count(w, C) + 1})
#
# Need to keep track of:
#   number of documents
#   number of documents in each class
#   total words for each class
#   each vocabulary word count for each class
n_doc = 0

# loop through corpora, tokenize and get necessary counts
# TODO: is this at all efficient?
def insert_into_vocab(word, category, vocab):
    try:
        vocab.loc[category, word] = 1 if pd.isna(vocab.loc[category, word]) else vocab.loc[category, word] + 1
    except KeyError:
        vocab.loc[category, word] = 1

test_vocab = pd.DataFrame()
insert_into_vocab('hello', 'cat1', test_vocab)
insert_into_vocab('world', 'cat2', test_vocab)
insert_into_vocab('hello', 'cat2', test_vocab)
insert_into_vocab('hello', 'cat2', test_vocab)
print(test_vocab, test_vocab.values)
