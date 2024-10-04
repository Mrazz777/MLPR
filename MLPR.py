import numpy as np
import matplotlib.pyplot as plt

amp_data = np.load('amp_data.npz')['amp_data']

# QUESTION 1a
fig, ax = plt.subplots(2,1)
'''
ax[0].plot(amp_data)
ax[0].set_title('Line Graph of Sequence in amp_data.npz')
ax[0].set_xlabel('Time Index')
ax[0].set_ylabel('Amplitude')

ax[1].hist(amp_data,bins=100)
ax[1].set_title('Histogram of Amplitudes in amp_data.npz')
ax[1].set_xlabel('Amplitude')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
'''

'''
R = len(amp_data) // 21
dataset = amp_data[0:R*21]
M = dataset.reshape(R,21)

'''

