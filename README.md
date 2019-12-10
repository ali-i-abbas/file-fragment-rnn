File Fragment Type Identification using Recurrent Neural Networks

This code is mostly a modification of code taken from "FiFTy: Large-scale File Fragment Type Identification using Neural Networks" avaiable at https://github.com/mittalgovind/fifty

To use this code follow this steps:

1- Download Scenario #1 (512-byte blocks) from http://dx.doi.org/10.21227/kfxw-8084 and unzip into data directory

2- Run utility.py to create the feature dataset at unigram folder

3- Run rnn_param.py to find the optimal hyperparameters. You can skip this step and go to next step, if you don't need to modify current best hyperparameters

4- Run rnn.py to train the network and get loss and accuracy plots, and also confusion matrix. The resulting model is saved as rnn.h5


An already saved model can be found in model directory.