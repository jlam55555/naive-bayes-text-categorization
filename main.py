import nltk

from naive_bayes import NaiveBayesClassifier

nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# init nltk tools
stemmer = PorterStemmer()
tokenizer = word_tokenize

# get in/out file information
train_labels_file = input('Enter training file: ')
test_file = input('Enter test file: ')
out_file = input('Enter out file: ')

# corpus files are relative to these files
path_to_train_labels_file = '/'.join(train_labels_file.split('/')[0:-1])
path_to_test_file = '/'.join(test_file.split('/')[0:-1])

# helper to get doc tokens
get_doc_tokens = lambda handle: \
    list(map(stemmer.stem, nltk.word_tokenize(''.join(handle.readlines()))))

# training stage
print('Beginning training stage...')
classifier = NaiveBayesClassifier()
with open(train_labels_file) as train_handle:
    for line in train_handle:
        doc_file, cls = line.rstrip('\n').split(' ')

        with open(f'{path_to_train_labels_file}/{doc_file}', 'r') as doc_handle:
            classifier.train(get_doc_tokens(doc_handle), cls)

classifier.compile()

# validation stage
print('Beginning validation stage...')
with open(test_file, 'r') as test_handle:
    doc_filenames = test_handle.read().splitlines()

    with open(out_file, 'w+') as out_handle:
        for doc_filename in doc_filenames:
            with open(f'{path_to_test_file}/{doc_filename}', 'r') as doc_handle:
                prediction = classifier.predict(get_doc_tokens(doc_handle))
            out_handle.write(f'{doc_filename} {prediction}\n')

print('Done.')
