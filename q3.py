import numpy as np

# Given matrix
X = np.array([
    [1, 4, 5],
    [1, 1, 0],
    [2, 3, 9]
])

# Step 1: Calculate the mean of each column (dimension)
column_means = np.mean(X, axis=0)

# Step 2: Subtract the column means from each corresponding element
X_centered = X - column_means
print(X_centered)

# Step 3: Perform SVD on the mean-centered data matrix
U, S, Vt = np.linalg.svd(X_centered)
