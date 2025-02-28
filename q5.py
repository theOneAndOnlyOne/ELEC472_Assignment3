import numpy as np
import matplotlib.pyplot as plt

# Time-series data
x = [1, 3, 4, 5, 0, 1, 11, -1, 3, -2, 5, 6]
t = np.arange(len(x))

# (a) Maximum, Window Size: 3, Overlap: 1
max_windows = []
max_positions = []
for i in range(0, len(x) - 2, 1):
    window = x[i:i+3]
    max_windows.append(max(window))
    max_positions.append(i+1)

# (b) Minimum, Window Size: 4, Overlap: 2
min_windows = []
min_positions = []
for i in range(0, len(x) - 3, 2):
    window = x[i:i+4]
    min_windows.append(min(window))
    min_positions.append(i+2)

# (c) Median and Mean, Window Size: 5, Overlap: 2
med_windows = []
mean_windows = []
med_positions = []
mean_positions = []
for i in range(0, len(x) - 4, 2):
    window = x[i:i+5]
    med_windows.append(np.median(window))
    mean_windows.append(np.mean(window))
    med_positions.append(i+2)
    mean_positions.append(i+2)

# (d) Range, Window Size: 3, Overlap: 0
range_windows = []
range_positions = []
for i in range(0, len(x) - 2, 3):
    window = x[i:i+3]
    range_windows.append(max(window) - min(window))
    range_positions.append(i+1)

# Print all features
print("Maximum Windows:", max_windows)
print("Maximum Positions:", max_positions)
print("Minimum Windows:", min_windows)
print("Minimum Positions:", min_positions)
print("Median Windows:", med_windows)
print("Mean Windows:", mean_windows)
print("Median Positions:", med_positions)
print("Mean Positions:", mean_positions)
print("Range Windows:", range_windows)
print("Range Positions:", range_positions)

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(t, x, label='Original Time-Series', marker='o')

# Maximum
plt.plot(max_positions, max_windows, label='Maximum (Win=3, Ov=1)', marker='x', linestyle='None')

# Minimum
plt.plot(min_positions, min_windows, label='Minimum (Win=4, Ov=2)', marker='s', linestyle='None')

# Median and Mean
plt.plot(med_positions, med_windows, label='Median (Win=5, Ov=2)', marker='^', linestyle='None')
plt.plot(mean_positions, mean_windows, label='Mean (Win=5, Ov=2)', marker='v', linestyle='None')

# Range
plt.plot(range_positions, range_windows, label='Range (Win=3, Ov=0)', marker='d', linestyle='None')

# Labels and Legend
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Time-Series and Extracted Features')
plt.legend()
plt.grid(True)
plt.show()