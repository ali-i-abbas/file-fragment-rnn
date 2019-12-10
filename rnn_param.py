
# code mostly taken from
# https://github.com/mittalgovind/fifty/blob/master/fifty/commands/train.py with modifications

import numpy as np
import os
import time
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras import callbacks, backend

from hyperopt import partial, Trials, fmin, hp, tpe, rand


start_time = time.time()

no_of_classes = 75
input_data_dir = './unigram/'

train_data = np.load(os.path.join(input_data_dir, 'train.npz'))
x_train, y_train = train_data['x'], train_data['y']
one_hot_y_train = to_categorical(y_train, no_of_classes)
print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape, one_hot_y_train.shape))

val_data = np.load(os.path.join(input_data_dir, 'val.npz'))
x_val, y_val = val_data['x'], val_data['y']
one_hot_y_val = to_categorical(y_val, no_of_classes)
print("Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, one_hot_y_val.shape))

# get best values for hyperparameters
def get_best():
    best_idx = df['accuracy'].idxmax()
    best = dict()
    best['shape'] = int(df['shape'].loc[best_idx])
    best['units'] = int(df['units'].loc[best_idx])
    best['layers'] = int(df['layers'].loc[best_idx])
    best['dense'] = int(df['dense'].loc[best_idx])
    return best

def train_network(parameters):
    print("\nParameters:")
    print(parameters)
    model = Sequential()

    try:
        # reshape training and validation data based on shape parameter
        x_t = x_train.reshape((x_train.shape[0], 1 * parameters['shape'], int(256 / parameters['shape'])))
        x_v = x_val.reshape((x_val.shape[0],x_t.shape[1],x_t.shape[2]))

        # Recurrent layer
        model.add(LSTM(parameters['units'], return_sequences=False, input_shape=(x_t.shape[1],x_t.shape[2])))

        # Dropout for regularization
        model.add(Dropout(0.1))

        # Fully connected layer
        for _ in range(parameters['layers']):
            model.add(Dense(parameters['dense']))
            model.add(LeakyReLU(alpha=0.3))

        # Output layer
        model.add(Dense(no_of_classes, activation='softmax'))

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, min_delta=0.01),
            callbacks.ModelCheckpoint('rnn.h5', monitor='val_accuracy'),
            callbacks.CSVLogger(filename='rnn.log', append=True)
        ]

        # Compile the model
        model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        history = model.fit(
            x=x_t,
            y=one_hot_y_train,
            epochs=32, batch_size=128, validation_data=(
                x_v, one_hot_y_val),
            verbose=2, callbacks=callbacks_list)
        loss = min(history.history['val_loss'])
        accuracy = max(history.history['val_accuracy'])
        backend.clear_session()
        parameters['accuracy'] = accuracy
        df.loc[len(df)] = list(parameters.values())
    except:
        accuracy = 0
        loss = np.inf

    print("Loss: {}".format(loss))
    print("Accuracy: {:.2%}".format(accuracy))
    return loss

df = pd.DataFrame(columns=['shape', 'units', 'layers', 'dense', 'accuracy'])

# these are the possible choices that will be evaluated to find the best combination of hyperparameters
# shape 1 corresponds to (1, 256), 2 to (2, 128), and 4 to (4, 64)
parameter_space = {
            'shape': hp.choice('shape', [1, 2, 4]),
            'units': hp.choice('units', [16, 32, 64, 128, 256, 512]),
            'layers': hp.choice('layers', [1, 2, 3]),
            'dense': hp.choice('dense', [32, 64, 128, 256, 512])
        }

trials = Trials()

# number of models that will be built and evaluated using the provided choices
max_evals = 225

algo = partial(
                tpe.suggest,
                n_EI_candidates=1000,
                gamma=0.2,
                n_startup_jobs=int(0.1 * max_evals),
            )

fmin(
    train_network,
    trials=trials,
    space=parameter_space,
    algo=algo,
    max_evals=max_evals,
    show_progressbar=False
)
df.to_csv('parameters.csv')
best = get_best()
print('\n-------------------------------------\n')

print('Hyper-parameter space exploration ended. \nRetraining the best again on the full dataset.')
train_network(best)
print('The best model has been retrained and saved as rnn.')

print("--- %s seconds ---" % (time.time() - start_time))

backend.clear_session()
