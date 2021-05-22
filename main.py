import util
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

if __name__ == '__main__':
    sequence = util.separe_column(r'monitoramento-cpu.txt', 'usr')
    train, test = util.split_sets(sequence, 0.70)

    n_steps = 3

    X, y = util.split_sequence(train.values.tolist(), n_steps)
    X_test, y_test = util.split_sequence(test.values.tolist(), n_steps)

    #converte os np array de string -> float
    X = X.astype(np.float)
    y = y.astype(np.float)
    X_test = X_test.astype(np.float)
    y_test = y_test.astype(np.float)

    #mudar axis do vetor
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    print(X.shape, X_test.shape, y.shape, y_test.shape)
    print(X[0], y[0])
    breakpoint()
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mse', 'acc'])

    history = model.fit(X, y, epochs=100, validation_data=(X_test, y_test), verbose=1)
