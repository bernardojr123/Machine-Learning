from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import itertools
from load_data import getXY
import numpy as np


algoritm = MLPClassifier(activation='tanh', solver='sgd', learning_rate='constant')
asdsa = algoritm.get_params()

x, y = getXY()

kf = KFold(10,shuffle=True)

#atributos = http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

gs = GridSearchCV(algoritm, param_grid={
    # 'learning_rate_init ': [0.5,0.1,0.05,0.01,0.005,0.001],
    # 'hidden_layer_sizes': [x for x in itertools.product((1,2,3),repeat=3)],
    'hidden_layer_sizes': [(1,1,1),(2,2,2),(3,3,3),(4,4,4)],
    'alpha' : [1e-04, 1e-05, 1e-06]})

folds = kf.split(x, y)

for k, (train, test) in enumerate(folds):
    x_train, x_validate, y_train, y_validate = train_test_split(
    x[train], y[train], test_size = 0.1)
    s = gs.fit(x_validate, y_validate).best_estimator_
    s.fit(x_train,y_train)
    predi = s.predict(x[test])
    contador = 0
    for i in range(len(predi)):
        if predi[i] == y[test][i]:
            contador += 1

    print("foi {} vezes".format(k))
    print("de {} acertou {}".format(len(predi),contador))
    break




