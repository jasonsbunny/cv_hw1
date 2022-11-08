from re import split
import numpy as np


tag=['scab','healthy','frog_eye_leaf_spot','rust','complex','powdery_mildew']


def onehot(data):
    labels=np.zeros((len(data),6))
    for i in range(len(data)):
        txt = split(' ', data[i])
        for j in range(len(tag)):
            for x in range(len(txt)):
                if txt[x] == tag[j]:
                    labels[i, j] = 1
    labels.astype(np.float32)
    return labels