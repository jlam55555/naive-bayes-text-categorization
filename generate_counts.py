# TODO: add description for this file
# TODO: add more debugging output
# this is in a separate file because it takes some time to read the corpora

import nltk
import numpy as np
import pickle

# required for word tokenizer
nltk.download('punkt')

# get training dataset parameters
train_labels_file = input('Enter training file: ')
pickle_file = input('Enter pickle outfile: ')

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

# save things
save_dict = {
    'n_doc': n_doc,
    'class_hist': class_hist,
    'token_index': token_index,
    'class_index': class_index,
    'vocab_array': vocab_array
}
with open(pickle_file, 'wb+') as pickle_file_handle:
    pickle.dump(save_dict, pickle_file_handle)
