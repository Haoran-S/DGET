import scipy.io as sio
import numpy as np

def read_data(file_name, num_samples, num_features ):
    labels = np.zeros((num_samples, 1))
    features = np.zeros((num_samples, num_features))
    f = open(file_name + ".txt", "r")
    for i in range(num_samples):
        cur = f.readline()
        cur = cur.split()
        labels[i] = cur[0]
        cur.pop(0)
        for item in cur:
            idx, val = item.split(':')
            features[i][int(idx)-1] = int(val)

    sio.savemat(file_name + ".mat", {'labels' : labels, 'features' : features})
    return 0

read_data("a9a_train", 32561, 123)
read_data("w8a_train", 49749, 300)
