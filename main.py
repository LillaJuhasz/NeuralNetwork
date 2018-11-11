import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy import stats
from pandas import DataFrame as df

ds = pd.read_csv('dataset.csv')
ds = ds.reset_index()

input_data = np.asarray(list((ds['input']).values))
input_float = pd.np.reshape([float(i) for i in input_data], (-1, 1))

output_quadratic = np.asarray(list((ds['output1']).values))
output_sin = np.asarray(list((ds['output2']).values))

activations = ['identity', 'logistic', 'relu']
solvers = ['sgd', 'adam', 'lbfgs']
hidden_layers = [(100, 1), (1000, 10, 1000, 10)]

regressor = list()
quadratic_scores = []
sin_scores = []
int_scores = []

counter = 0
for i in activations:
    for j in solvers:
        for k in hidden_layers:
            regressor.append(MLPRegressor(activation=i, solver=j, hidden_layer_sizes=k, random_state=1, max_iter=1000))

            regressor[counter].fit(input_float, output_quadratic)
            try:
                score = regressor[counter].score(input_float, output_quadratic)
            except ValueError:
                score = 'Unsuccessful'
            pred = regressor[counter].predict(input_float)

            print('quadratic', score, i, j, k, sep='; ', file=open('scores.txt', 'a'))
            if score != "Unsuccessful":
                quadratic_scores.append(score)
                int_scores.append(score)

            plt.plot(input_float, output_quadratic, 'o')
            plt.plot(input_float, pred, 'o')
            plt.savefig("plots/"+j + '' + i + '' + str(hidden_layers.index(k)) + 'quad.pdf')
            plt.show()

            counter += 1

counter = 0
for i in activations:
    for j in solvers:
        for k in hidden_layers:
            regressor.append(MLPRegressor(activation=i, solver=j, hidden_layer_sizes=k, random_state=1, max_iter=1000))

            regressor[counter].fit(input_float, output_sin)
            try:
                score = regressor[counter].score(input_float, output_sin)
            except ValueError:
                score = 'Unsuccessful'
            pred = regressor[counter].predict(input_float)

            print('sin', score, i, j, k, sep='; ', file=open('scores.txt', 'a'))
            if score != "Unsuccessful":
                sin_scores.append(score)
                int_scores.append(score)

            plt.plot(input_float, output_sin, 'o')
            plt.plot(input_float, pred, 'o')
            plt.savefig("plots/"+j + '' + i + '' + str(hidden_layers.index(k)) + 'sin.pdf')
            plt.show()

            counter += 1

print('quadratic', stats.describe(quadratic_scores), sep='; ', file=open('stats.txt', 'a'))
print('sin', stats.describe(sin_scores), sep='; ', file=open('stats.txt', 'a'))

quadratic_scores.append(0.0)
data = df(data={'quadratic': quadratic_scores, 'sin': sin_scores})
print(data.describe(), sep='; ', file=open('stats.txt', 'a'))
