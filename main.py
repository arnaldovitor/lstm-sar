from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import util
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

if __name__ == '__main__':
    sequence = util.separe_column(r'monitoramento-cpu.txt', 'usr')

    n_steps = 3

    X, y = util.split_sequence(sequence.values.tolist(), n_steps)

    #converte os np array de string -> float
    X = X.astype(np.float)
    y = y.astype(np.float)


    #mudar axis do vetor
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))


    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mse'])

    history = model.fit(X, y, epochs=20, verbose=1)

    resultados = model.evaluate(X, y)

    pred = model.predict(X)

    print("loss, mse, acc = ", resultados)

    for i in range(len(pred)):
        print(pred[i], y[i])



plt.plot(y, label = 'Comparação entre Y_true e Y_pred', color = 'blue')
plt.plot(pred, label = 'Y_pred', color = 'red')
plt.xlabel('Número de neurônios na hidden layer.')
plt.title("n_steps: " + str(n_steps))
plt.legend()
plt.show()