# Given a set of counts, perform text categorization
# TODO: make the whole model into its own class, or generally refactor

import nltk
import pickle
import numpy as np
import functools

# required for word_tokenizer
nltk.download('punkt')

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

# TODO: maybe move this to the other file
# class conditionals: perform +1 smoothing and convert counts to log-likelihoods
vocab_array += 1
log_likelihoods = np.log10(vocab_array / np.sum(vocab_array, axis=0))

# class priors: convert counts to log-likelihoods
log_priors = np.log10(np.array(list(class_hist.values())) / n_doc)[np.newaxis, :]

# categorize a directory of files
test_file = input('Enter test file: ')
out_file = input('Enter out file: ')

with open(test_file, 'r') as test_file_handle:
    test_filenames = test_file_handle.read().splitlines()

# corpus files are relative to the train labels file, not necessarily this script directory
path_to_test_file = '/'.join(test_file.split('/')[0:-1])

class_likelihoods = np.zeros(shape=(len(test_filenames), len(class_hist)))
for i, document_filename in enumerate(test_filenames):
    with open(f'{path_to_test_file}/{document_filename}', 'r') as document_handle:
        tokens = nltk.tokenize.word_tokenize(''.join(document_handle.readlines()))

    # remove tokens that are not in the original vocabulary
    tokens = filter(lambda token: token in token_index, tokens)

    # calculate log-likelihood of being in each class
    class_likelihoods[i, :] = functools.reduce(lambda acc, token: acc + log_likelihoods[token_index[token], :], tokens, log_priors)

likely_class_indices = np.argmax(class_likelihoods, axis=1)

# human-readable format
class_names = list(class_hist.keys())
likely_classes = [class_names[likely_class_index] for likely_class_index in likely_class_indices]

with open(out_file, 'w+') as out_file_handle:
    for filename, likely_class in zip(test_filenames, likely_classes):
        out_file_handle.write(f'{filename} {likely_class}\n')
