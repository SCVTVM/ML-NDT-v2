import os
import numpy as np
from keras.models import load_model

# Paths
model_path = "../training_variants/modelcpntndt_nn_sigm_relu.keras"
model_path = "../training_variants/modelcpnt1c8caf1a-456c-4ce6-aa9f-af4678be622d.keras"
# model_path = "modelcpntndt_nn_just_relu.keras"
validation_folder = "../data/validation"
training_folder = "../data/training"

def load_labels(labels_file):
    return np.loadtxt(labels_file, dtype=np.float32)

def preprocess_data(file_path):
    data = np.fromfile(file_path, dtype=np.uint16).astype('float32')
    data -= data.mean()
    data /= data.std() + 0.0001
    return np.reshape(data, (-1, 256, 256, 1), 'C')

def process_dataset(folder, model, combined_results, single_output):
    for file in os.listdir(folder):
        if file.endswith(".bins"):
            data_path = os.path.join(folder, file)
            labels_path = data_path.replace(".bins", ".labels")

            # Load data and labels
            data = preprocess_data(data_path)
            labels = load_labels(labels_path)
            binary_labels = labels[:, 0]
            regression_labels = labels[:, 1]

            # Make predictions
            if single_output:
                predictions = model.predict(data)
                binary_predictions = predictions > 0.5
                regression_predictions = predictions

            else:  # Two outputs
                binary_predictions, regression_predictions = model.predict(data)

            # Post-process predictions
            binary_predictions_bin = binary_predictions > 0.5
            regression_predictions_filt = regression_predictions.copy()
            regression_predictions_filt[~binary_predictions_bin] = 0

            if np.sum(np.abs(binary_predictions_bin.flatten() - binary_labels)):
                print(np.sum(binary_predictions_bin))
                print(np.sum(binary_labels))
                print( np.sum(np.abs(binary_predictions_bin.flatten() - binary_labels)))
                print( np.sum(binary_predictions_bin.flatten() - binary_labels))

            # Accumulate results
            combined_results['binary_predicted'] += np.sum(binary_predictions_bin)
            combined_results['binary_labels'] += np.sum(binary_labels)
            combined_results['binary_residuals'] += np.sum(np.abs(binary_predictions_bin.flatten() - binary_labels))

            combined_results['regression_predicted'] += np.sum(regression_predictions_filt)
            combined_results['regression_labels'] += np.sum(regression_labels)
            combined_results['regression_residuals'] += np.sum(np.abs(regression_predictions_filt.flatten() - regression_labels))


# Initialize combined results
combined_results = {
    'binary_predicted': 0,
    'binary_labels': 0,
    'binary_residuals': 0,
    'regression_predicted': 0,
    'regression_labels': 0,
    'regression_residuals': 0,
}

# Load model
model = load_model(model_path)
example_input = np.random.random((1, 256, 256, 1)).astype('float32')
single_output = model.predict(example_input).shape[-1] == 1

# Process validation datasets
process_dataset(validation_folder, model, combined_results, single_output)
# Print combined summary
print("\nCombined Summary of just validation results:")
print(f"Sum of Binary Predictions: {combined_results['binary_predicted']}")
print(f"Sum of Binary Labels: {combined_results['binary_labels']}")
print(f"Sum of Binary Residuals: {combined_results['binary_residuals']:.2f}")

print(f"Sum of Regression Predictions: {combined_results['regression_predicted']:.2f}")
print(f"Sum of Regression Labels: {combined_results['regression_labels']:.2f}")
print(f"Sum of Regression Residuals: {combined_results['regression_residuals']:.2f}")

# Process training + validation datasets
# process_dataset(training_folder, model, combined_results)


# Process validation datasets


# Process training + validation datasets
process_dataset(training_folder, model, combined_results, single_output)

# Print combined summary
print("\nCombined Summary of all of the data:")
print(f"Sum of Binary Predictions: {combined_results['binary_predicted']}")
print(f"Sum of Binary Labels: {combined_results['binary_labels']}")
print(f"Sum of Binary Residuals: {combined_results['binary_residuals']:.2f}")

print(f"Sum of Regression Predictions: {combined_results['regression_predicted']:.2f}")
print(f"Sum of Regression Labels: {combined_results['regression_labels']:.2f}")
print(f"Sum of Regression Residuals: {combined_results['regression_residuals']:.2f}")