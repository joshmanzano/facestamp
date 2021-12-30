import pickle
import numpy as np
import matplotlib.pyplot as plt

a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

try:
    saved_heatmap_data = pickle.load(open('heatmap_data.bin','rb'))
    saved_heatmap_data_raw = pickle.load(open('heatmap_data_raw.bin','rb'))

    print(saved_heatmap_data.shape)

except:
    print('exit')

