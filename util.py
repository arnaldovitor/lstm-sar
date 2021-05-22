import pandas as pd
from numpy import array

#splita a coluna em steps, ex: X: [1, 2, 3] e y: [4]
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#separa só uma coluna do dataframe e troca ',' por '.'
def separe_column(input_path, column):
    df = pd.read_csv(input_path, sep=' ')
    df = df.apply(lambda x: x.str.replace(',','.'))
    data = pd.DataFrame(df)
    sequence = data[column]
    return sequence

#divide a sequência em treino e teste
def split_sets(sequence, train_perc):
    train_size = int(len(sequence) * train_perc)
    train, test = sequence[0:train_size], sequence[train_size:len(sequence)]
    return train, test