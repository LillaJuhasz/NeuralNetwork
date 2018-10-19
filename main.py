import csv
import sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier, MLPRegressor

df = pd.read_csv('dataset.csv')
y = list((df['output']).values)

X = list((df['input']).values)
X_f= pd.np.reshape([float(i) for i in X], (-1, 1))

clf = MLPRegressor(solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(100), random_state=1)

clf.fit(X_f, y)
pred=clf.predict(X_f)
print(pred)
plt.plot(X_f, y, 'o')
plt.plot(X_f, pred, 'o')
plt.show()

print(clf.score(X_f, y))