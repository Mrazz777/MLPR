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



# Question 3b

def Phi(C, K):
    '''Constructs a C x K design matrix, representing the C
    most recent time steps before the time we wish to predict (t=1).
    The row for time t has K features.'''
    # Construct input time vector
    l_t = t[20-C:20,0].reshape(-1,1)
    M = np.ones((C,1))

    # Construct design matrix
    if K == 1:
        return M
    else:
        for i in range(1,K):
            M = np.hstack((M, l_t**(i))) # Add columns of t^i
        return M


def make_vv(C, K):
    '''Returns the vector v for a model with K features and context of C previous amplitudes.'''
    Phi_1 = np.ones((K, 1)) # First row of Phi
    d_m = Phi(C, K)
    a = np.linalg.inv(np.dot(d_m.T, d_m)) # Inverse of d_m.T * d_m
    v = np.dot(np.dot(d_m, a.T), Phi_1)  # d_m * a * Phi_1
    return v

# QUESTION 3biii

v_linear = make_vv(20, 2)
predict_linear = np.dot(v_linear.T, X_shuf_train[0])

v_quartic = make_vv(20, 5)
predict_quartic = np.dot(v_quartic.T, X_shuf_train[0])

# Check if the predictions are the same as the last values of the polynomial fits to 8 decimal places
print(np.isclose(y_linear[-1], predict_linear[0], atol=1e-8))
print(np.isclose(y_quartic[-1], predict_quartic[0], atol=1e-8))

print("predict_linear", predict_linear[0])
print("predict_quartic", predict_quartic[0])

'''Returns:
True
True
predict_linear 0.043741808439555926
predict_quartic 0.03998262151479061'''


# QUESTION 3ci
# Initialise variables
C = 20
K = 4
M_e = np.ones((C,K))

for i in range(0,C): #row
    for j in  range(0,K):
        # Require c > k so that the matrix is lower triangular
        if i > j :
            error = (np.dot( X_shuf_train[0,(19-i):] , make_vv(i+1,j+1) )-y_shuf_train[0])
            M_e[i,j] = error**2
        else:
            M_e[i,j] = np.inf

print(M_e)

min_e_s = np.min(M_e)
print(min_e_s)
print(np.where(M_e == min_e_s))
print(M_e[11,1])

'''Smallest square error is 3.725290298457679e-09 for C=12 and K=2.'''


# QUESTION 3cii

C = 12
K = 2
beta_t = make_vv(C,K)
def mse(X_shuf_set, y_shuf_set, beta_t):
    '''Calculates mean squared error for given dataset subset.'''
    l = len(y_shuf_set)
    mse = 0
    for i in range(0,l):
        mse += (np.dot(X_shuf_set[i,19-11:],beta_t)-y_shuf_set[i])**2
    return mse/l

# Find MSE for training, validation and test sets
mse_train = mse(X_shuf_train, y_shuf_train, beta_t)[0]
mse_val = mse(X_shuf_val, y_shuf_val, beta_t)[0]
mse_test = mse(X_shuf_test, y_shuf_test, beta_t)[0]
print("Training set MSE: ", mse_train)
print("Validation set MSE: ", mse_val)
print("Test set MSE: ", mse_test)

'''Returns:
Training set MSE:  0.00026760607554292954
Validation set MSE:  0.0002659203590978971
Test set MSE:  0.0002626577603204292
'''

for i in range(1,20) :
    eee = 0
    beta_t = make_vv(i+1,2)
    for j in range(0,len(y_shuf_val)):
        eee += (np.dot(X_shuf_val[j,19-i:],beta_t)-y_shuf_val[j])**2
    print(eee/l_train)
