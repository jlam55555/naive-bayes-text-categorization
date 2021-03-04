# TODO: implement mp version of this

import numpy as np
from typing import Collection

class NaiveBayesClassifier:
    """
    Perform Naive Bayes bag-of-words text categorization with Laplace smoothing.
    This class does not perform tokenization, and it expects the full set of
    tokens (the vocabulary) and class labels to be known beforehand.

    Equations for Naive Bayes:
    - P(C) = N_C / N_doc
    - P(w_i|C) = (count(w_i, C) + 1) / (\sum_{w\in V}{count(w, C) + 1})

    Need to keep track of:
    - number of documents
    - number of documents in each class
    - total words for each class
    - each vocabulary word count for each class
    """

    def __init__(self, vocab_set: Collection[str], class_set: Collection[str]):
        """
        Naive Bayes text categorizer

        :param vocab:   collection of (all, unique) string tokens
        :param classes: collection of (all, unique) string document classes
        """

        # manage counts of tokens for class-conditional probabilities
        self.vocab_array = np.zeros(shape=(len(vocab_set), len(class_set)),
                                    dtype=np.uintc)

        # string -> index mappings
        self.token_index = {word: i for i, word in enumerate(vocab_set)}
        self.class_index = {clss: i for i, clss in enumerate(class_set)}
        self.class_list = list(class_set)

        # manage category and total document counts for class priors
        self.doc_count = 0
        self.class_doc_counts = np.zeros(shape=(len(class_set), ))

        # when built
        self.log_likelihoods = self.log_priors = None

        self.built = False

    def train_document(self, tokens: Collection[str], document_class: str):
        """
        Trains on a document

        :param tokens:          collection of string tokens in a document
        :param document_class:  class label of the current document
        :return:                None
        """
        assert not self.built, 'model must be trained before build'

        class_index = self.class_index[document_class]
        for token in tokens:
            self.vocab_array[self.token_index[token], class_index] += 1

        # update document counts for class priors
        self.doc_count += 1
        self.class_doc_counts[class_index] += 1

    def build_model(self):
        """
        Computes class-conditional token probabilities and class priors from
        training counts for use at inference-time

        :return:    None
        """
        assert not self.built, 'model is already built'

        # perform +1 (Laplace) smoothing
        self.vocab_array += 1

        # convert class-conditionals to log-likelihoods (base-10 is arbitrary)
        self.log_likelihoods = np.log10(self.vocab_array /
                                        np.sum(self.vocab_array, axis=0))

        # convert prior counts to log-likelihoods
        self.log_priors = np.log10(self.class_doc_counts /
                                   self.doc_count)[np.newaxis, :]

        self.built = True

    def predict(self, tokens: Collection[str]) -> str:
        """
        Predict class of a document given its tokens

        :param tokens:  collection of string tokens in a document
        :return:        predicted class label for the document
        """
        assert self.built, 'model must be built before prediction'

        likelihoods = np.array(self.log_priors)
        for token in filter(lambda token: token in self.token_index, tokens):
            likelihoods += self.log_likelihoods[self.token_index[token], :]

        return self.class_list[np.argmax(likelihoods, axis=1)[0]]
