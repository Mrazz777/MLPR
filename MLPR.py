import numpy as np
import matplotlib.pyplot as plt

amp_data = np.load('amp_data.npz')['amp_data']

# QUESTION 1a
#fig, ax = plt.subplots(2,1)
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


# QUESTION 1c

# Obtain lstsq coefficients
t = np.linspace(0,19/20,20).reshape(-1,1)
d_M_linear = np.hstack([np.ones((20, 1)), t])
d_M_quartic = np.hstack([np.ones((20, 1)), t, t ** 2, t ** 3, t ** 4])
w_linear = np.linalg.lstsq(d_M_linear, X_shuf_train[0].reshape(-1, 1), rcond=None)[0]
w_quartic = np.linalg.lstsq(d_M_quartic, X_shuf_train[0].reshape(-1, 1), rcond=None)[0]

# Find coords for the linear and quartic fits
t_plot = np.linspace(0, 1, 100)
y_linear = w_linear[0] + w_linear[1] * t_plot
y_quartic = w_quartic[0] + w_quartic[1] * t_plot + w_quartic[2] * t_plot ** 2 + w_quartic[3] * t_plot ** 3 + w_quartic[4] * t_plot ** 4

# Plot the data points and the fits
fig1,ax1 = plt.subplots(1,1)
ax1 = plt.scatter(t, X_shuf_train[0])
ax1 = plt.scatter(1, y_shuf_train[0], c="r")
ax1 = plt.plot(t_plot, y_linear, linewidth=2, label="Polynomial Fit (Linear)")
ax1 = plt.plot(t_plot, y_quartic, linewidth=2, label="Polynomial Fit (Quartic)")
ax1 = plt.title("Polynomial Fits of Degree 1 and 4 on One Row of Training Data")
ax1 = plt.xlabel("Time Index")
ax1 = plt.ylabel("Amplitude")
ax1 = plt.legend()
plt.show()

def phi(C,K):
    l_t = t[20-C:20,0].reshape(-1,1)
    M = np.ones((C,1))
    if K == 1:
        return M
    else:
        for i in range(1,K):
            M = np.hstack((M,l_t**(i)))
        return M
print(phi(3,4))