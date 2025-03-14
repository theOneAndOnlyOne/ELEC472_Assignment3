import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, medfilt, firwin, filtfilt

# Given time series
x = np.array([2, 1, 3, 2, 1, 7, 3, 4, 2, 2, 1, 3, 3, 4, 6, 7, 3, 2, 2, 3, 3, 2])

# (a) Moving average of size 5
def moving_average(signal, size):
    return np.convolve(signal, np.ones(size)/size, mode='valid')

moving_avg = moving_average(x, 5)
#print(moving_avg)
# (b) FIR filter with h = [0, 1, 2, 1, 0]
def apply_fir_filter(x, h):
    """
    Applies an FIR filter to the input signal x using filter coefficients h.
    
    Parameters:
    x (list or numpy array): Input signal.
    h (list or numpy array): Filter coefficients.
    
    Returns:
    numpy array: Filtered output signal.
    """
    x = np.array(x)
    h = np.array(h)
    
    output_length = len(x) - len(h) + 1  # Output length based on valid convolution
    if output_length <= 0:
        raise ValueError("Filter length should be less than or equal to input length.")
    
    y = np.zeros(output_length)
    
    for t in range(output_length):
        for k in range(len(h)):
            y[t] += x[t + k] * h[k]
    
    return y
h = np.array([0, 1, 2, 1, 0])
fir_filter = np.array([2.25, 2, 2.75, 4.5, 4.25, 3.25, 2.5, 1.75, 1.75, 2.5, 3.25, 4.25, 5.75, 5.75, 3.75, 2.25, 2.25,
2.75])
#print(fir_filter)
# (c) Median filter of size 5
median_filter = medfilt(x, kernel_size=5)
median_filter = median_filter[2:-2]   # Offset to include the first 5 numbers in x
#print(median_filter)
# (d) IIR filter y(t) = 0.3 * x(t) + 0.7 * y(t-1), y(0) = 0
y = np.zeros_like(x, dtype=float)
y[0] = 0.3*x[0]
for t in range(1, len(x)):
    #print("x:", x[t])
    y[t] = 0.3 *x[t] + 0.7 * y[t-1]
    #print("y:", y[t])
#print(y)
y = np.concatenate(([0], y[:-1]))  # Adjusting for the reduced length



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, label='Original Signal', marker='o')
plt.plot(range(4, len(moving_avg) + 4), moving_avg, label='Moving Average (size 5)', marker='x')
plt.plot(range(4, len(fir_filter) + 4), fir_filter, label='FIR Filter [0, 1, 2, 1, 0]', marker='s')
plt.plot(range(4, len(median_filter) + 4), median_filter, label='Median Filter (size 5)', marker='^')
plt.plot(y, label='IIR Filter', marker='d')


plt.legend()
plt.title('Time Series and Filtered Outputs')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# High-pass version of (a) Moving Average (size 5)
# High-pass filter is calculated by subtracting the moving average from the original signal
highpass_moving_avg = x[2:-2] - moving_avg  # Adjusting for the reduced length
print(highpass_moving_avg)
# High-pass version of (b) FIR Filter
# Design a high-pass FIR filter with similar cutoff frequency
highpass_fir = x[2:-2] - fir_filter  # Adjusting for the reduced length
print(highpass_fir)

# Plotting high-pass filters
plt.figure(figsize=(10, 6))
plt.plot(x, label='Original Signal', marker='o')
plt.plot(range(2, len(highpass_moving_avg) + 2), highpass_moving_avg, label='High-pass Moving Average', marker='x')
plt.plot(range(2, len(highpass_fir) + 2), highpass_fir, label='High-pass FIR Filter', marker='s')
plt.legend()
plt.title('High-pass Filtered Outputs')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Display numerical outputs for both high-pass filters

