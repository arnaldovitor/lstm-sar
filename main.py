from matplotlib import pyplot as plt
from util import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout

if __name__ == '__main__':
    sequence = separe_column(r"monitoramento-disco.txt", 'usado')
    train, test = split_sets(sequence, 0.8)

    n_steps = 4

    X, y = split_sequence(train.values.tolist(), n_steps)
    X_test, y_test = split_sequence(test.values.tolist(), n_steps)
    y_true = np.concatenate((y, y_test), axis=0)

    #converte os np array de string -> float
    X = X.astype(np.float)
    y = y.astype(np.float)
    X_test = X_test.astype(np.float)
    y_test = y_test.astype(np.float)

    #mudar axis do vetor
    n_features = 1
    n_seq = 2
    n_steps = 2
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))


    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mape', 'mse', 'mae'])

    history = model.fit(X, y, validation_data=(X_test, y_test), epochs=100, verbose=1)
    pred_x = model.predict(X)
    pred_x_test = model.predict(X_test)

    dif = (len(y_true) - len(pred_x_test))
    eixo_x_test = [(i + dif) for i in range(len(pred_x_test))]

    plt.plot(y_true, label='Y_true', color='blue')
    plt.plot(pred_x, label='Y_pred', color='red', linestyle='-.')
    plt.plot(eixo_x_test, pred_x_test, label='Y_test_pred', color='green', linestyle='-.')
    plt.title("n_steps: " + str(n_steps))
    plt.legend()
    plt.show()


