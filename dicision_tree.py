from sklearn import tree
from sklearn.model_selection import KFold
from load_data import getXY

algoritm = tree.DecisionTreeClassifier()

x, y = getXY()

kf = KFold(10,shuffle=True)

folds = kf.split(x,y)

for k, (train, test) in enumerate(folds):
    algoritm.fit(x[train], y[train])
    y_pred = algoritm.predict(x[test])
    acertos = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[test][i]:
            acertos += 1
    print("Dos {} casos, o algoritmo acertou {} vezes".format(len(y[test]), acertos))

