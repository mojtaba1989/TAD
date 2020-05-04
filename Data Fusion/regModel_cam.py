from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FILE_NAME = '01-Oct-2019-14-04'
NEW = True

# if __name__ == '__main__':
#
#     is_error = False
#     for idx, arg in enumerate(sys.argv):
#         if arg in ['--file', '-f']:
#             FILE_NAME = str(sys.argv[idx+1])
#             del sys.argv[idx]
#             del sys.argv[idx]
#         else:
#             is_error = True
#
#     for idx, arg in enumerate(sys.argv):
#         if arg in ['--new', '-n']:
#             NEW = True
#             del sys.argv[idx]
#         else:
#             is_error = True
#
#     if len(sys.argv) != 1:
#         is_error = True
#     else:
#         for arg in sys.argv:
#             if arg.startswith('-'):
#                 is_error = True


CWD_PATH = os.getcwd()
PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'TPDA-lined-cleaned-' + FILE_NAME + '.csv')
PATH_TO_MODEL = os.path.join(CWD_PATH, 'Angle_Detection_Model', 'angle_detection_estimator.h5')


raw_dataset = pd.read_csv(PATH_TO_CSV,
                      na_values="?", comment='\t',
                      sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()

dataset["New_Ver"] = dataset["New_Ver"].shift(-20)
dataset = dataset[["T_est", "d_T_est", "T_est_inv", "New_Ver"]]
dataset = dataset.dropna()

data = dataset.copy()
labels = data.pop("New_Ver")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('New_Ver')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('New_Ver')
test_labels = test_dataset.pop('New_Ver')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_data = norm(data)

def TAD_loss(labels, predictions):
    abs_r = tf.keras.backend.abs(labels-predictions)
    max_error = tf.keras.backend.abs(labels) / 10 + .25
    loss = tf.keras.backend.mean(tf.keras.backend.less_equal(abs_r, max_error))
    return loss


def build_model():
    if NEW:
          model = keras.Sequential([
            layers.Dense(1024, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
          ])
    else:
        model = tf.keras.models.load_model(PATH_TO_MODEL)


    optimizer =tf.keras.optimizers.Adam(.001)


    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse', TAD_loss])
    return model

model = build_model()

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(labels)
line2, = ax1.plot(labels * 0.5)
plt.show()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 5 == 0:
        line2.set_ydata(model.predict(normed_data))
        plt.pause(0.01)
        print('')

    print('.', end='')



EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=100)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plt.waitforbuttonpress()

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()




def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [vernier]')
    plt.plot(hist['epoch'], hist['TAD_loss'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_TAD_loss'],
            label = 'Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Sample')
    plt.ylabel('Residuals')
    plt.scatter(data.index, model.predict(normed_data).reshape(-1, 1)-np.array(labels).reshape(-1, 1)
              , label='Residuals')
    plt.legend()

    plt.figure()
    plt.xlabel('Angle')
    plt.ylabel('Residuals')
    plt.scatter(np.array(labels).reshape(-1, 1), model.predict(normed_data).reshape(-1, 1)-np.array(labels).reshape(-1, 1)
              , label='Residuals')
    plt.legend()


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$vernier^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.legend()
    plt.show(block=False)




plot_history(history)
plt.waitforbuttonpress()
model.save(PATH_TO_MODEL)

