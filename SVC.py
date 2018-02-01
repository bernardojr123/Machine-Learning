from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import itertools
from load_data import getXY
import numpy as np

algoritm = svm.SVC()

tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

clf = GridSearchCV(algoritm, tuned_parameters, cv=5)




