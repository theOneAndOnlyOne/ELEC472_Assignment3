import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, medfilt

# Given time series
x = np.array([2, 1, 3, 2, 1, 7, 3, 4, 2, 2, 1, 3, 3, 4, 6, 7, 3, 2, 2, 3, 3, 2])

# (a) Moving average of size 5
def moving_average(signal, size):
    return np.convolve(signal, np.ones(size)/size, mode='valid')

moving_avg = moving_average(x, 5)
print(moving_avg)
# (b) FIR filter with h = [0, 1, 2, 1, 0]
h = np.array([0, 1, 2, 1, 0])
fir_filter = np.convolve(x, h, mode='same')
print(fir_filter)
# (c) Median filter of size 5
median_filter = medfilt(x, kernel_size=5)
print(median_filter)
# (d) IIR filter y(t) = 0.3 * x(t) + 0.7 * y(t-1), y(0) = 0
y = np.zeros_like(x, dtype=float)
y[0] = 0.3*x[0]
for t in range(1, len(x)):
    print("x:", x[t])
    y[t] = 0.3 *x[t] + 0.7 * y[t-1]
    print("y:", y[t])
print(y)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, label='Original Signal', marker='o')
plt.plot(range(4, len(moving_avg) + 4), moving_avg, label='Moving Average (size 5)', marker='x')
plt.plot(fir_filter, label='FIR Filter [0, 1, 2, 1, 0]', marker='s')
plt.plot(median_filter, label='Median Filter (size 5)', marker='^')
plt.plot(y, label='IIR Filter', marker='d')

plt.legend()
plt.title('Time Series and Filtered Outputs')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

