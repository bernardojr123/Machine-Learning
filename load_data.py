from sklearn.datasets import fetch_mldata

def getXY():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'] , mnist['target']
    return X, y

