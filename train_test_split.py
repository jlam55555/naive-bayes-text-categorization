"""
Something akin to sklearn.model_selection.train_test_split, modified for the
purposes of this assignment. Here we don't do any tuning so we don't really
need a tuning/validation/dev dataset, and we only split into train and test
in order to evaluate our performance.

This takes in a training file (*.labels) and spits out three subset files:
- a smaller training set w/ labels: (*.sub.labels)
- a list of test documents w/o labels: (*.test.list)
- a list of test documents w/ labels: (*.test.labels)
"""
from random import random

# get input file
train_file = input('Enter input training file: ')

# generate default filenames for output files
path_components = train_file.split('/')
path = '/'.join(path_components[:-1])
filename = path_components[-1]
# trim .labels and _train if present
if filename.endswith('.labels'):
    filename = filename[:-7]
if filename.endswith('_train'):
    filename = filename[:-6]
default_sub_train_filename = f'{path}/{filename}_sub.labels'
default_test_list_filename = f'{path}/{filename}_test.list'
default_test_labels_filename = f'{path}/{filename}_test.labels'

# prompt for output filenames
sub_train_filename = input(f'Training subset filename [{default_sub_train_filename}]: ')\
                     or default_sub_train_filename
test_list_filename = input(f'Test list filename [{default_test_list_filename}]: ')\
                     or default_test_list_filename
test_labels_filename = input(f'Test labels filename [{default_test_labels_filename}]: ')\
                       or default_test_labels_filename

# get train_test_split
default_split = 0.8
split = input(f'Train/test split [0.8]: ') or default_split

# perform split
with open(train_file, 'r') as train_file_handle, \
     open(sub_train_filename, 'w+') as sub_train_handle, \
     open(test_list_filename, 'w+') as test_list_handle, \
     open(test_labels_filename, 'w+') as test_labels_handle:

    for line in train_file_handle:
        if random() < split:
            sub_train_handle.write(line)
        else:
            test_list_handle.write(line.split(' ')[0] + '\n')
            test_labels_handle.write(line)
