import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
from utils import *

def generateds(path, txt, format, history):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    signals = []
    labels =[]
    for content in contents:
        value = content.split()
        label_ = value[1:]
        label = []
        for h in range(len(label_)):
            a = int(label_[h])
            label.append(a)

        vib_path = path + value[0] + format
        vib = pd.read_csv(vib_path,encoding='UTF-8')
        signals_set = vib.iloc[2000:3600, 2:].values

        for i in range(0, len(signals_set)):
            if i % history == 0:
                signals.append(signals_set[i:i+history, :])
                labels.append(label)

        del label

    signal, label_a = np.array(signals), np.array(labels)
    label_a = np.squeeze(label_a)
    return signal, label_a

'''
def ADJ():
    path = './vib_data/adj.csv'
    adj=pd.read_csv(path)
    adj=adj.iloc[:, 0].values
    adj=np.array(adj, dtype=np.int32)
    adj_map = {j: i for i, j in enumerate(adj)}

    edges=
    print(adj_map)
ADJ()

path='./vib_data/time step data/'
txt='./vib_data/position/position.txt'
format='.csv'
history=400
#x_train, y_train,x_valid, y_valid,x_test, y_test= generateds(path, txt, format, history)
X , y = generateds(path, txt, format, history)
print(X.shape)
print(type(y[1]))
print((y[1]))
'''