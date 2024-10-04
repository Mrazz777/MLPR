import numpy as np
import matplotlib.pyplot as plt

amp_data = np.load('amp_data.npz')['amp_data']

fig, ax = plt.subplots(2,1)
'''
ax[0].plot(amp_data)
ax[1].hist(amp_data,bins=100)
plt.show()
'''

'''
R = len(amp_data) // 21
dataset = amp_data[0:R*21]
M = dataset.reshape(R,21)

'''

