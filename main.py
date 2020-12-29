"""
Driver for Naive Bayes training and testing, including tokenization. The main
Naive Bayes algorithm is in naive_bayes.py.
"""

# required for word tokenizer
import nltk
nltk.download('punkt')

from naive_bayes import NaiveBayesClassifier

# get dataset parameters
train_labels_file = input('Enter training file: ')
test_file = input('Enter test file: ')
out_file = input('Enter out file: ')

# corpus files are relative to these files
path_to_train_labels_file = '/'.join(train_labels_file.split('/')[0:-1])
path_to_test_file = '/'.join(test_file.split('/')[0:-1])

### TRAINING STAGE ###
class_set = set()
document_class_map = {}

# parse training labels
with open(train_labels_file) as training_file_handle:
    for line in training_file_handle:
        file, class_name = line.rstrip('\n').split(' ')
        class_set.add(class_name)
        document_class_map[file] = class_name

# build vocab first to prevent resizing
# this step doesn't take too long so it's not terrible to make it completely
# separate from the tokenization
vocab_set = set()
for document_filename in document_class_map.keys():
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        vocab_set.update(nltk.tokenize.word_tokenize(''.join(document_handle.readlines())))

# the local naive bayes classifier, not the nltk one with the same name
nb_classifier = NaiveBayesClassifier(vocab_set, class_set)

# main training loop: loop through corpora, tokenize and get necessary counts
for document_filename, document_class in document_class_map.items():
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        tokens = nltk.tokenize.word_tokenize(''.join(document_handle.readlines()))
    nb_classifier.train_document(tokens, document_class)

# convert counts to log probabilities to prepare for inference
nb_classifier.build_model()

### TESTING STAGE ###
# get filenames from test file
with open(test_file, 'r') as test_file_handle:
    test_filenames = test_file_handle.read().splitlines()

predictions = []
for i, document_filename in enumerate(test_filenames):
    with open(f'{path_to_test_file}/{document_filename}', 'r') as document_handle:
        tokens = nltk.tokenize.word_tokenize(''.join(document_handle.readlines()))
    predictions.append((document_filename, nb_classifier.predict(tokens)))

# output to file
with open(out_file, 'w+') as out_file_handle:
    for filename, likely_class in predictions:
        out_file_handle.write(f'{filename} {likely_class}\n')
