from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import util
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

if __name__ == '__main__':
    sequence = separe_column(r"monitoramento-disco.txt", 'usado')
    train, test = split_sets(sequence, 0.8)

    n_steps = 4

    X, y = split_sequence(train.values.tolist(), n_steps)
    X_test, y_test = split_sequence(test.values.tolist(), n_steps)

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
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['acc', 'mse'])

    history = model.fit(X, y, epochs=150, verbose=1)

    resultados = model.evaluate(X, y)

    pred = model.predict(X)

    print("loss, mse, acc = ", resultados)

    for i in range(len(pred)):
        print(pred[i], y[i])

    plt.plot(y, label = 'Y_true', color = 'blue')
    plt.plot(pred, label = 'Y_pred', color = 'red')
    plt.title("n_steps: " + str(n_steps))
    plt.legend()
    plt.show()

