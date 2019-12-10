import os
from collections import Counter

import numpy as np

# code mostly taken from make_new_dataset function in
# https://github.com/mittalgovind/fifty/blob/master/fifty/commands/train.py with modifications
def make_unigram_dataset():
    # download Scenario #1 (512-byte blocks) from http://dx.doi.org/10.21227/kfxw-8084 and unzip into data directory
    input_data_dir = './data/'

    # this directory will hold the unigram feature data that will be used for training
    out_data_dir = './unigram/'

    alphabet_size = 256	# number of values that a byte can hold

    total_size = 7680000 # total number of samples in train.npz, val.npz, and test.npz combined

    train_data = np.load(os.path.join(input_data_dir, 'train.npz'))
    x_t, y_train = train_data['x'], train_data['y']

    # get 30% of total dataset for training
    n_samples = int(total_size * 0.3)
    x_t = x_t[:n_samples]
    y_train = y_train[:n_samples]

    # get unigram of training data
    x_train = unigram(x_t, alphabet_size)

    np.savez_compressed(os.path.join(out_data_dir, 'train.npz'),
                        x=x_train, y=y_train)
    del train_data, x_t, x_train, y_train

    val_data = np.load(os.path.join(input_data_dir, 'val.npz'))
    x_v, y_val = val_data['x'], val_data['y']
    # we use all validation and testing data, because they are already contain 10% of total sample size each
    x_val = unigram(x_v, alphabet_size)

    np.savez_compressed(os.path.join(out_data_dir, 'val.npz'),
                        x=x_val, y=y_val)
    del val_data, x_v, x_val, y_val

    test_data = np.load(os.path.join(input_data_dir, 'test.npz'))
    xt, y_test = test_data['x'], test_data['y']
    x_test = unigram(xt, alphabet_size)

    np.savez_compressed(os.path.join(out_data_dir, 'test.npz'),
                        x=x_test, y=y_test)
    del test_data, xt, x_test, y_test

def unigram(array, alphabet_size):
    u = np.empty((array.shape[0], alphabet_size))
    for i, val in enumerate(array):
        c = Counter(val)
        for j in range(alphabet_size):
            u[i, j] = c[j]
    return u

make_unigram_dataset()


