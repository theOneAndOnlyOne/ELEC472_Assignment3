# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Given time series
x = np.array([2, 1, 3, 2, 1, 7, 3, 4, 2, 2, 1, 3, 3, 4, 6, 7, 3, 2, 2, 3, 3, 2])

# Original FIR Filter (b) with h = [0, 1, 2, 1, 0]
original_fir_h = np.array([0, 1, 2, 1, 0])
original_fir_output = np.convolve(x, original_fir_h, mode='same')

# Original IIR Filter (d) y(t) = 0.3 * x(t) + 0.7 * y(t-1)
y_original_iir = np.zeros_like(x, dtype=float)
y_original_iir[0] = 0.3 * x[0]  # Start at index 0

for t in range(1, len(x)):
    y_original_iir[t] = 0.3 * x[t] + 0.7 * y_original_iir[t-1]

# (b) Modified FIR Filter with h = [0.2, 0.4, 0.8, 0.4, 0.2]
modified_fir_h = np.array([0.2, 0.4, 0.8, 0.4, 0.2])
modified_fir_output = np.convolve(x, modified_fir_h, mode='same')

# (d) Modified IIR Filter y(t) = 0.2 * x(t) + 0.8 * y(t-1)
y_modified_iir = np.zeros_like(x, dtype=float)
y_modified_iir[0] = 0.2 * x[0]  # Start at index 0

for t in range(1, len(x)):
    y_modified_iir[t] = 0.2 * x[t] + 0.8 * y_modified_iir[t-1]

# Plotting all filters on the same graph
plt.figure(figsize=(12, 8))
plt.plot(x, label='Original Signal', marker='o')
plt.plot(original_fir_output, label='Original FIR Filter', marker='s')
plt.plot(y_original_iir, label='Original IIR Filter', marker='d')
plt.plot(modified_fir_output, label='Modified FIR Filter', marker='x')
plt.plot(y_modified_iir, label='Modified IIR Filter', marker='^')
plt.legend()
plt.title('Original and Modified Low-Pass Filter Outputs')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Display numerical outputs for both modified filters
modified_fir_output, y_modified_iir
