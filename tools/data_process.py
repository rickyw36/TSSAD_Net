import numpy as np
file = "train.npy"
data = np.load(file)
data = data.transpose()
for i in range(data.shape[0]):
    np.save('signal' + str(i), data[i])
