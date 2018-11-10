import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy import stats

df = pd.read_csv('dataset.csv')
df = df.reset_index()
X = np.asarray(list((df['input']).values))
X_f = pd.np.reshape([float(i) for i in X], (-1, 1))

y_quadratic = np.asarray(list((df['output1']).values))
y_sin = np.asarray(list((df['output2']).values))

activations = ['identity', 'logistic', 'relu']
solvers = ['sgd', 'adam', 'lbfgs']
hidden_layers = [(100, 1), (1000, 10, 1000, 10)]

clf = list()
quadratic_scores = []
sin_scores = []
int_scores = []

a = 0
for i in activations:
    for j in solvers:
        for k in hidden_layers:
            clf.append(MLPRegressor(activation=i, solver=j, hidden_layer_sizes=k, random_state=1, max_iter=1000))

            clf[a].fit(X_f, y_quadratic)
            try:
                score = clf[a].score(X_f, y_quadratic)
            except ValueError:
                score = 'Unsuccessful'

            print('quadratic', score, i, j, k, sep='; ', file=open('scores.txt', 'a'))
            if score != "Unsuccessful":
                quadratic_scores.append(score)
                int_scores.append(score)

            pred = clf[a].predict(X_f)
            plt.plot(X_f, y_quadratic, 'o')
            plt.plot(X_f, pred, 'o')
            plt.savefig("plots/"+j + '' + i + '' + str(hidden_layers.index(k)) + 'quad.pdf')
            plt.show()

            a += 1

a = 0
for i in activations:
    for j in solvers:
        for k in hidden_layers:
            clf.append(MLPRegressor(activation=i, solver=j, hidden_layer_sizes=k, random_state=1, max_iter=1000))

            clf[a].fit(X_f, y_sin)
            try:
                score = clf[a].score(X_f, y_sin)
            except ValueError:
                score = 'Unsuccessful'

            print('sin', score, i, j, k, sep='; ', file=open('scores.txt', 'a'))
            if score != "Unsuccessful":
                sin_scores.append(score)
                int_scores.append(score)

            pred = clf[a].predict(X_f)
            plt.plot(X_f, y_sin, 'o')
            plt.plot(X_f, pred, 'o')
            plt.savefig("plots/"+j + '' + i + '' + str(hidden_layers.index(k)) + 'sin.pdf')
            plt.show()

            a += 1

print('quadratic', stats.describe(quadratic_scores), sep='; ', file=open('stats.txt', 'a'))
print('sin', stats.describe(sin_scores), sep='; ', file=open('stats.txt', 'a'))
