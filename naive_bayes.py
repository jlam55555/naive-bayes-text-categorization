from collections import defaultdict
from typing import Collection
from math import log


class NaiveBayesClassifier:

    def __init__(self, smoothing_factor=0.05) -> None:
        # for training/inference
        self.vocab = set()

        # for training
        self.counts = defaultdict(lambda: defaultdict(int))
        self.classes = defaultdict(int)
        self.sf = smoothing_factor

        # for inference
        self.llikelihoods = {}
        self.lpriors = defaultdict(float)
        self.compiled = False

    def train(self, tokens: Collection[str], cat: str) -> None:
        assert not self.compiled, 'model is already compiled'

        for token in tokens:
            self.vocab.add(token)
            self.counts[cat][token] += 1

        self.classes[cat] += 1

    def compile(self) -> None:
        assert not self.compiled, 'model is already compiled'

        # calculate log P(w|C)
        vocab_size = len(self.vocab)
        for cls, tokens in self.counts.items():
            cls_token_sum = sum(tokens.values()) + vocab_size * self.sf
            cls_default_smoothed = log(self.sf / cls_token_sum)
            self.llikelihoods[cls] = defaultdict(lambda: cls_default_smoothed)
            for token, count in tokens.items():
                self.llikelihoods[cls][token] = \
                    log((count + self.sf) / cls_token_sum)

        # calculate log P(C)
        cls_doc_sum = sum(self.classes.values())
        for cls, doc_count in self.classes.items():
            self.lpriors[cls] = log(doc_count / cls_doc_sum)

        # destroy old count data
        self.counts = self.classes = None
        self.compiled = True

    def predict(self, tokens: Collection[str]) -> str:
        assert self.compiled, 'model is not compiled'

        # discard tokens not in the training set
        tokens = list(filter(lambda token: token in self.vocab, tokens))

        best_score, best_cls = float('-inf'), ''
        for cls in self.lpriors:
            score = self.lpriors[cls]

            for token in tokens:
                score += self.llikelihoods[cls][token]

            if score > best_score:
                best_score = score
                best_cls = cls

        assert best_cls != '', 'could not predict best class'
        return best_cls
