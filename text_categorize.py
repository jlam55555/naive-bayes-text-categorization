# Given a set of counts, perform text categorization
# TODO: make the whole model into its own class, or generally refactor

import nltk
import os
import pickle
import numpy as np

# pickle_file = input('Counts pickle file: ')
pickle_file = 'counts/corpus1.pkl'

# read in counts file
with open(pickle_file, 'rb') as pickle_file_handle:
    save_dict = pickle.load(pickle_file_handle)
n_doc = save_dict['n_doc']
class_hist = save_dict['class_hist']
token_index = save_dict['token_index']
class_index = save_dict['class_index']
vocab_array = save_dict['vocab_array']

# class conditionals: perform +1 smoothing and convert counts to log-likelihoods
vocab_array += 1
vocab_array = np.log10(vocab_array / np.sum(vocab_array, axis=0))

# class priors: convert counts to log-likelihoods
for class_name, class_count in class_hist.items():
    class_hist[class_name] = np.log10(class_count / n_doc)
