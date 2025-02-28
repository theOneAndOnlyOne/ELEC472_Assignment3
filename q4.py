import numpy as np
import pandas as pd

# Original sequences with NaN represented as np.nan
X1 = np.array([1, 2, np.nan, 3, 3, np.nan, 0, 2, -2, -2, 1, 5, np.nan, 9])
X2 = np.array([np.nan, -1, 2, np.nan, -2, -1, 3, 2, -2, np.nan, -4, 1])

# Function to calculate standard deviation ignoring NaN
def calculate_std(sequence):
    return np.nanstd(sequence)

# Calculate standard deviation before imputation
std_X1_before = calculate_std(X1)
std_X2_before = calculate_std(X2)

# 1. Zero-Padding
X1_zero = np.nan_to_num(X1, nan=0)
X2_zero = np.nan_to_num(X2, nan=0)

# 2. Sample and Hold (Forward Fill)
X1_sample = pd.Series(X1).fillna(method='ffill').to_numpy()
X2_sample = pd.Series(X2).fillna(method='ffill').to_numpy()

# 3. Linear Interpolation
X1_interp = pd.Series(X1).interpolate().to_numpy()
X2_interp = pd.Series(X2).interpolate().to_numpy()

# Calculate standard deviations after imputation
std_X1_zero = np.std(X1_zero)
std_X2_zero = np.std(X2_zero)

std_X1_sample = np.std(X1_sample)
std_X2_sample = np.std(X2_sample)

std_X1_interp = np.std(X1_interp)
std_X2_interp = np.std(X2_interp)

# Calculate change in standard deviation
change_X1_zero = abs(std_X1_zero - std_X1_before)
change_X2_zero = abs(std_X2_zero - std_X2_before)

change_X1_sample = abs(std_X1_sample - std_X1_before)
change_X2_sample = abs(std_X2_sample - std_X2_before)

change_X1_interp = abs(std_X1_interp - std_X1_before)
change_X2_interp = abs(std_X2_interp - std_X2_before)

# Compile results into a DataFrame for better visualization
results = pd.DataFrame({
    'Method': ['Zero-Padding', 'Sample and Hold', 'Interpolation'],
    'X1 Std Change': [change_X1_zero, change_X1_sample, change_X1_interp],
    'X2 Std Change': [change_X2_zero, change_X2_sample, change_X2_interp]
})

results
