import numpy as np
import matplotlib.pyplot as plt

name = "results_ndt_nn_relu_relu"
# name = "results_ndt_nn_sigm_relu"
name = "results_ndt_nn_tanh"

# Load the data from the text file
file_name = '../training_variants/' + name + '.txt'  # Replace custom_name if needed
res = np.loadtxt(file_name)

# Plot the data
plt.scatter(res[:, 1], res[:, 3], color='blue', label='Regression Predictions')
plt.scatter(res[:, 1], res[:, 2], color='red', label='Binary Predictions')
plt.xlabel('Ground Truth Values (ys[:, 1])')
plt.ylabel('Predictions')
plt.legend()
plt.grid(True)  # Optional: show grid
plt.title('Prediction Scatter Plot')
plt.savefig('../training_variants/' + name + '.png')
plt.show()