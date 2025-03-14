import numpy as np

def calculate_std_without_nan(data):
    """
    Compute the standard deviation while ignoring NaN values.
    Formula: sigma = sqrt( (1/N) * sum((xi - mean)^2) )
    """
    valid_data = data[~np.isnan(data)]  # Remove NaN values
    std_dev = np.std(valid_data, ddof=1)
    return std_dev

def zero_padding_imputation(data):
    """Replace NaN values with 0."""
    return np.nan_to_num(data, nan=0)

def sample_and_hold_imputation(data):
    """Replace NaN values with the last valid observation."""
    data_filled = data.copy()
    for i in range(len(data_filled)):
        if np.isnan(data_filled[i]):
            data_filled[i] = data_filled[i - 1] if i > 0 else 0  # Assume 0 if first element is NaN
    return data_filled

def linear_interpolation_imputation(data):
    """Interpolate missing values linearly."""
    indices = np.arange(len(data))
    mask = np.isnan(data)
    data[mask] = np.interp(indices[mask], indices[~mask], data[~mask])
    return data

# Sample dataset with NaN values
data = np.array([1, 2, np.nan, 3, 3, np.nan, 0, 2, -2, -2, 1, 5, np.nan, 9])
data =  np.array([np.nan, -1,2, np.nan, -2, -1,3,2, -2, np.nan, -4,1])
# Standard deviation before imputation
std_before = calculate_std_without_nan(data)

# Apply imputation techniques
data_zero_padded = zero_padding_imputation(data.copy())
data_sample_hold = sample_and_hold_imputation(data.copy())
data_interpolated = linear_interpolation_imputation(data.copy())

# Standard deviation after imputation
std_after_zero_padded= calculate_std_without_nan(data_zero_padded)
std_after_sample_hold = calculate_std_without_nan(data_sample_hold)
std_after_interpolated = calculate_std_without_nan(data_interpolated)

# Print results
print("Standard Deviation Calculation:")
print(f"Before Imputation: {std_before:.4f}")
print(f"After Zero Padding: {std_after_zero_padded:.4f}")
print(f"After Sample & Hold: {std_after_sample_hold:.4f}")
print(f"After Linear Interpolation: {std_after_interpolated:.4f}")
# Calculate and print the change in standard deviation
print("\nChange in Standard Deviation:")
print(f"Zero Padding: {std_after_zero_padded - std_before:.4f}")
print(f"Sample & Hold: {std_after_sample_hold - std_before:.4f}")
print(f"Linear Interpolation: {std_after_interpolated - std_before:.4f}")