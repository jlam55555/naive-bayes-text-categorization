# TODO: add description for this file
# TODO: add more debugging output

import nltk
import numpy as np
import functools

# required for word tokenizer
nltk.download('punkt')

# get dataset parameters
train_labels_file = input('Enter training file: ')
test_file = input('Enter test file: ')
out_file = input('Enter out file: ')

### TRAINING STAGE ###
# from training labels, get list of all classes
# each class will store its own document count
class_hist = {}
document_class_map = {}

# parse training labels
with open(train_labels_file) as training_file_handle:
    for line in training_file_handle:
        file, class_name = line.rstrip('\n').split(' ')
        class_hist[class_name] = 0
        document_class_map[file] = class_name

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

# corpus files are relative to the train labels file, not necessarily this script directory
path_to_train_labels_file = '/'.join(train_labels_file.split('/')[0:-1])

# build vocab first to prevent resizing
# this step doesn't take too long so it's not terrible to make it completely
# separate from the tokenization
# TODO: allow multithreading for both this and the next section
vocab_set = set()
for document_filename in document_class_map.keys():
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        vocab_set.update(nltk.tokenize.word_tokenize(''.join(document_handle.readlines())))

# simple df-like array (string indices, but homogeneous uint type)
vocab_array = np.zeros(shape=(len(vocab_set), len(class_hist)), dtype=np.uintc)
token_index = {word: i for i, word in enumerate(vocab_set)}
class_index = {class_name: i for i, class_name in enumerate(class_hist.keys())}

# loop through corpora, tokenize and get necessary counts
for document_filename, document_class in document_class_map.items():
    # add document to appropriate counts
    n_doc += 1
    class_hist[document_class] += 1

    # tokenize document, add words to vocab array
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        for token in nltk.tokenize.word_tokenize(''.join(document_handle.readlines())):
            vocab_array[token_index[token], class_index[document_class]] += 1

# class conditionals: perform +1 smoothing and convert counts to log-likelihoods
vocab_array += 1
log_likelihoods = np.log10(vocab_array / np.sum(vocab_array, axis=0))

# class priors: convert counts to log-likelihoods
log_priors = np.log10(np.array(list(class_hist.values())) / n_doc)[np.newaxis, :]

### TESTING STAGE ###
with open(test_file, 'r') as test_file_handle:
    test_filenames = test_file_handle.read().splitlines()

# corpus files are relative to the test file, not necessarily this script directory
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

# output to file
with open(out_file, 'w+') as out_file_handle:
    for filename, likely_class in zip(test_filenames, likely_classes):
        out_file_handle.write(f'{filename} {likely_class}\n')
