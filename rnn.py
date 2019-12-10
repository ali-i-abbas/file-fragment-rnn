# code mostly taken from
# https://github.com/mittalgovind/fifty/blob/master/fifty/commands/train.py with modifications
# run rnn with best hyperparameters

import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from itertools import product

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras import callbacks, backend

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix


# taken with modification from scikit-learn ConfusionMatrixDisplay class
def plot(confusion_matrix, display_labels, title, include_values=True, cmap='viridis',
         xticks_rotation='horizontal', values_format=None, ax=None):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=50)
    matplotlib.rc('ytick', labelsize=50)
    fig, ax = plt.subplots(figsize=(70, 70))

    cm = confusion_matrix
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)
        if values_format is None:
            values_format = '0.2f'

        # print text with appropriate color depending on background
        thresh = (cm.max() - cm.min()) / 2.
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text_[i, j] = ax.text(j, i,
                                  format(cm[i, j], values_format),
                                  ha="center", va="center",
                                  color=color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(im_, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=70)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels
           )

    ax.set_xlabel(xlabel="Predicted label", fontsize=80, labelpad=100)
    ax.set_ylabel(ylabel="True label", fontsize=80, labelpad=100)

    ax.set_ylim((n_classes - 0.5, -0.5))
    ax.set_title(title, fontsize=100, pad=100)
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


start_time = time.time()

labels = ['jpg', 'arw', 'cr2', 'dng', 'gpr', 'nef', 'nrw', 'orf', 'pef', 'raf', 'rw2', '3fr', 'tiff', 'heic', 'bmp',
          'gif', 'png', 'ai', 'eps', 'psd', 'mov', 'mp4', '3gp', 'avi', 'mkv', 'ogv', 'webm', 'apk', 'jar', 'msi',
          'dmg', '7z', 'bz2', 'deb', 'gz', 'pkg', 'rar', 'rpm', 'xz', 'zip', 'exe', 'mach-o', 'elf', 'dll', 'doc',
          'docx', 'key', 'ppt', 'pptx', 'xls', 'xlsx', 'djvu', 'epub', 'mobi', 'pdf', 'md', 'rtf', 'txt', 'tex',
          'json', 'html', 'xml', 'log', 'csv', 'aiff', 'flac', 'm4a', 'mp3', 'ogg', 'wav', 'wma', 'pcap', 'ttf',
          'dwg', 'sqlite']

no_of_classes = 75
input_data_dir = './unigram/'

train_data = np.load(os.path.join(input_data_dir, 'train.npz'))
x_train, y_train = train_data['x'], train_data['y']
x_train = x_train.reshape((x_train.shape[0], 4, 64))
one_hot_y_train = to_categorical(y_train, no_of_classes)
print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape, one_hot_y_train.shape))

val_data = np.load(os.path.join(input_data_dir, 'val.npz'))
x_val, y_val = val_data['x'], val_data['y']
x_val = x_val.reshape((x_val.shape[0], x_train.shape[1], x_train.shape[2]))
one_hot_y_val = to_categorical(y_val, no_of_classes)
print("Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, one_hot_y_val.shape))

batch_size = 128
dense = 64
units = 512
epochs = 32
lambda_param = 0.11

model = Sequential()

# Recurrent layer
model.add(LSTM(units, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))

# Dropout for regularization
model.add(Dropout(0.1))

# Fully connected layer
model.add(Dense(dense))
model.add(LeakyReLU(alpha=0.3))

# Output layer
model.add(Dense(no_of_classes, activation='softmax'))

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, min_delta=0.01),
    callbacks.ModelCheckpoint('rnn.h5', monitor='val_accuracy'),
    callbacks.CSVLogger(filename='rnn.log', append=True)
]

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    x=x_train,
    y=one_hot_y_train,
    epochs=epochs, batch_size=batch_size, validation_data=(
        x_val, one_hot_y_val),
    verbose=2, callbacks=callbacks_list)

loss = min(history.history['val_loss'])
accuracy = max(history.history['val_accuracy'])

print("Loss: {}".format(loss))
print("Accuracy: {:.2%}".format(accuracy))

print('\nModel Performance: Log Loss and Accuracy on test data')

test_data = np.load(os.path.join(input_data_dir, 'test.npz'))
x_test, y_test = test_data['x'], test_data['y']
x_test = x_test.reshape((x_test.shape[0], x_train.shape[1], x_train.shape[2]))
one_hot_y_test = to_categorical(y_test, no_of_classes)
print("Testing Data loaded with shape: {} and labels with shape - {}".format(x_test.shape, one_hot_y_test.shape))
r = model.evaluate(x_test, one_hot_y_test, batch_size=batch_size, verbose=2)

test_loss = r[0]
test_accuracy = r[1]

print(f'Loss: {round(test_loss, 4)}')
print(f'Accuracy: {round(100 * test_accuracy, 2)}%')

print("--- %s seconds ---" % (time.time() - start_time))

plt.clf()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("rnn_accuracy.png", type="png", dpi=300)

plt.clf()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig("rnn_loss.png", type="png", dpi=300)

plt.clf()

y_test_pred = model.predict_classes(x_test)
cm1 = confusion_matrix(y_test, y_test_pred)

cm2 = confusion_matrix(y_test, y_test_pred, normalize='true')

# save the confusion matrices
np.savez_compressed('rnn_cm.npz', cm1 = cm1, cm2 = cm2)

title = "Normalized confusion matrix"
plot(confusion_matrix=cm2, display_labels=labels, title=title, include_values=True, cmap=plt.cm.Blues,
     xticks_rotation='vertical')

plt.savefig("rnn_confusion_matrix.png", type="png", dpi=100)

backend.clear_session()
