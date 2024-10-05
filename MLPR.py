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
'''In the line graph, there are small bursts of regions with high amplitude,
unevenly distributed throughout the sequence. There are 4 main clusters of these bursts.
The histogram shows a steep bell-shaped distribution centred around 0:
amplitudes close to 0 are most frequent.'''

# QUESTION 1b

# Split the data into 21-length sequences
R = len(amp_data) // 21
dataset = amp_data[0:R*21]
M = dataset.reshape(R,21)

# Shuffle rows then find num records for the training, validation and test sets
np.random.seed(123)
rdataset = np.random.permutation(M)
l_train = np.floor(R*0.7).astype(int)
l_val = np.floor(R * 0.85).astype(int)
#l_test = R-l_train-l_vail.astype(int)

# Split the data into training, validation and test sets
X_shuf_train = rdataset[0:l_train , 0:20]
y_shuf_train = rdataset[0:l_train,20]
X_shuf_val = rdataset[l_train:l_val, 0:20]
y_shuf_val = rdataset[l_train:l_val, 20]
X_shuf_test = rdataset[l_val:, 0:20]
y_shuf_test = rdataset[l_val:, 20]

t = np.linspace(0,19/20,20).reshape(-1,1)
#print(type(t))
#print(t)
d_M = np.hstack((np.ones((20,1)),t))
d_M_q = np.hstack((np.ones((20,1)),t,t**2,t**3,t**4))
beta_f = np.linalg.lstsq(d_M,X_shuf_train[0].reshape(-1,1),rcond=None)[0]
beta_f_q = np.linalg.lstsq(d_M_q,X_shuf_train[0].reshape(-1,1),rcond=None)[0]

yf = np.dot(d_M,beta_f)
print(d_M.shape)
print(yf)
#fig1,ax1 = plt.subplots(1,1)

#ax1 = plt.scatter(t,X_shuf_test[0])
#ax1 = plt.scatter(1,y_shuf_test[0],c="r")
#ax1 = plt.plot(t,yf,linewidth=2)
#plt.show()