# naive-bayes-text-categorization
Text categorization using Naive Bayes for ECE467 project 1

### Model description
The text categorization is performed using a Naive Bayesian bag-of-words model. Tokenization is performed using NLTK's `word_tokenizer` utility.

### Generating train/test splits
The files in `corpora/corpus[1-3]/*`, `analyze.pl`, and `corpus[1-3]_train.labels` were provided from [the course webpage][1]. `corpora/corpus1_test.list` and `corpora/corpus1_test.labels` were also provided to demonstrate the format of the test files and test labels.

The `*.labels` files include both filenames and class labels (for the training dataset and for evaluating the test dataset), and the `*.list` files include only document filenames (for evaluation).

Random train/test splits for corpora 2 and 3 (`corpora/corpus[2-3]_sub.labels`, `corpora/corpus[2-3]_test.list`, and `corpora/corpus[2-3]_test.labels`), to mimick the train/test structure of corpus 1, were created using `train_test_split.py`:

```bash
$ python3 train_test_split.py
Enter input training file: corpora/corpus3_train.labels
Training subset filename [corpora/corpus3_sub.labels]: 
Test list filename [corpora/corpus3_test.list]: 
Test labels filename [corpora/corpus3_test.labels]: 
```

### Training and prediction
These train/test splits can be used to train a Naive Bayes classifier and predict on the test set:
```bash
$ python3 main.py
Enter training file: corpora/corpus3_sub.labels
Enter test file: corpora/corpus3_test.list
Enter out file: predictions/corpus3_pred.labels
```

### Model evaluation
A confusion matrix is generated using `analyze.pl` on the predicted and true class labels:
```bash
$ perl analyze.pl predictions/corpus3_pred.labels corpora/corpus3_test.labels
Processing answer file...
Found 6 categories: Sci Ent Fin Spo Wor USN
Processing prediction file...

150 CORRECT, 25 INCORRECT, RATIO = 0.857142857142857.

CONTINGENCY TABLE:
        Sci     Ent     Fin     Spo     Wor     USN     PREC
Sci     20      0       0       0       0       0       1.00
Ent     0       1       0       0       0       0       1.00
Fin     1       2       19      0       0       0       0.86
Spo     0       0       0       14      0       0       1.00
Wor     0       1       0       1       53      3       0.91
USN     5       4       2       3       3       43      0.72
RECALL  0.77    0.12    0.90    0.78    0.95    0.93    

F_1(Sci) = 0.869565217391304
F_1(Ent) = 0.222222222222222
F_1(Fin) = 0.883720930232558
F_1(Spo) = 0.875
F_1(Wor) = 0.929824561403509
F_1(USN) = 0.811320754716981
```

[1]: http://faculty.cooper.edu/sable2/courses/spring2020/ece467/
