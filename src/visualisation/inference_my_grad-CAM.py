
from __future__ import print_function

import tensorflow as tf
from keras.models import Model
import keras


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from scipy.ndimage import zoom

model_path = "my_ndt_nn_v1.keras"

model_path = "../training_variants/modelcpntndt_nn_tanh.keras"
# model_path = "modelcpntndt_nn_relu_relu.keras"
# model_path = "modelcpnt1c8caf1a-456c-4ce6-aa9f-af4678be622d.keras"
# model_path = "modelcpntndt_nn_sigm_relu.keras"
# model_path = "modelcpntndt_nn_just_relu.keras"
data_path = "../../data/validation/F68B8BC9-C4D5-4848-923E-A68176F821D2.bins"
results_path = "../../data/validation/F68B8BC9-C4D5-4848-923E-A68176F821D2.labels"

dont_filter = 1


def plot_with_labels_and_explanations(input_image, heatmap, binary_label, regression_label):
    plt.figure(figsize=(10, 10))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image[0, :, :, 0], cmap='gray')
    plt.title(f'Input Image\nBinary Label: {binary_label}, Regression Label: {regression_label:.2f}')
    plt.axis('off')

    # Grad-CAM heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(input_image[0, :, :, 0], cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.title('Grad-CAM Heatmap\n(Key Regions Highlighted)')
    plt.axis('off')

    # Add color gradient legend
    cbar_ax = plt.axes([0.15, 0.1, 0.7, 0.02])  # Colorbar axis (x, y, width, height)
    ColorbarBase(cbar_ax, cmap='jet', norm=Normalize(vmin=0, vmax=1), orientation='horizontal')
    cbar_ax.set_title('Contribution\nIntensity', fontsize=10)


    plt.tight_layout()
    plt.show()


def load_labels(labels_file):
    return np.loadtxt(labels_file, dtype=np.float32)



model = keras.models.load_model(model_path)

# Load and preprocess input data
rxs = np.fromfile(data_path, dtype=np.uint16).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std() + 0.0001
rxs = np.reshape(rxs, (-1, 256, 256, 1), 'C')


labels = load_labels(results_path)
binary_labels = labels[:, 0]
regression_labels = labels[:, 1]


# if len(model.predict(rxs)) == 2:
#
#     # Get predictions (binary and regression)
#     binary_predictions, regression_predictions = model.predict(rxs)
#
#     binary_predictions_bin = binary_predictions > 0.5
#     regression_predictions_filt = regression_predictions.copy()
#     regression_predictions_filt[~binary_predictions_bin] = 0
#
#
#
#     binary_res = np.abs(binary_predictions_bin.flatten() - binary_labels)
#     regression_res = np.abs(regression_predictions_filt.flatten() - regression_labels)
#
#     # Display results
#     print("Binary Predictions vs. Labels:")
#     print(np.column_stack((binary_predictions, binary_predictions_bin.flatten(), binary_labels)))
#     print("Absolute Difference (Binary):")
#     print(binary_res)
#
#     print("\nRegression Predictions vs. Labels:")
#     print(np.column_stack((regression_predictions, regression_predictions_filt.flatten(), regression_labels)))
#     print("Absolute Difference (Regression):")
#     print(regression_res)
#
#     # Summarize results
#     print("\nSummary:")
#     print("Sum of Binary Predictions:", sum(binary_predictions_bin))
#     print("Sum of Binary Labels:", sum(binary_labels))
#     print("Sum of Binary Residuals:", sum(binary_res))
#
#     print("Sum of Regression Predictions:", sum(regression_predictions_filt))
#     print("Sum of Regression Labels:", sum(regression_labels))
#     print("Sum of Regression Residuals:", sum(regression_res))
#
# else:
#     predictions = model.predict(rxs)
#     print(predictions)
#     # labels = load_labels(results_path)
#     # labels = np.array([labels[:, 0]]).T
#     res = np.abs(predictions - binary_labels)
#     print(np.hstack((predictions, labels)))
#
#     print(res)
#     print(sum(predictions))
#     print(sum(labels))
#     print(sum(res))


def grad_cam(model, input_image, target_layer_name, target_class_idx = None):
    # Create a sub-model up to the target layer
    target_layer = model.get_layer(target_layer_name)
    grad_model = Model(inputs=model.input, outputs=[target_layer.output, model.output])

    input_image_tens = tf.convert_to_tensor(input_image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(input_image)
        if target_class_idx is None:
            target_class = predictions
        else:
            target_class = predictions[target_class_idx]

    # Modify target_class if it's too close to 1.0

    grads = tape.gradient(target_class, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    # Weight the feature maps with the pooled gradients
    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(pooled_grads * feature_maps, axis=-1)

    # Normalize the heatmap
    heatmap = norm_heatmap(heatmap)

    return heatmap, predictions

def norm_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)
    # print(np.max(heatmap))
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, input_image, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Apply colormap
    # heatmap_r = tf.image.resize(heatmap, (input_image.shape[1], input_image.shape[2]))
    heatmap_r = tf.image.resize(heatmap, (input_image.shape[0], input_image.shape[1]))

    # overlay = heatmap_r * alpha + input_image / input_image.max()  # Scale input for visualization
    overlay = heatmap_r
    return overlay


# # Load an input image and preprocess it
# image_idx = 2  # Choose an image index
# input_image = rxs[image_idx:image_idx + 1]  # Single image batch
#
# print(model.summary())
#
# # Generate Grad-CAM heatmap
# heatmap, predictions = grad_cam(model, input_image, target_class_idx=1, target_layer_name='conv2d_1')  # Change target layer name
#
# # Overlay heatmap on the original input image
# overlay = overlay_heatmap(heatmap, input_image[0])
# print(overlay.shape)
#
#
# # Extract labels for the current image
# binary_label = labels[image_idx, 0]
# regression_label = labels[image_idx, 1]


def plot_heatmaps_for_layers(input_image, model, binary_label, regression_label, image_idx):
    # Define layer names and target class indices
    filt_num = 1

    layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']

    # Create a larger figure for the grid
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(2, 9)

    # Create the image subplot that spans the first 2x2 area
    ax = fig.add_subplot(gs[:2, :2])

    # Plot the original image in the top-left (larger area)
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    ax.set_title(f'Original Image\nBinary Label: {binary_label}, Regression Label: {regression_label:.2f}')
    ax.axis('off')

    # Initialize list to store heatmaps for averaging later
    heatmaps_class_0 = []
    heatmaps_class_1 = []

    # Generate and plot Grad-CAM heatmaps for each layer for target_class_idx=0
    for idx, layer_name in enumerate(layer_names):
        ax = fig.add_subplot(gs[0, idx + 2])
        heatmap, predictions = grad_cam(model, input_image, target_class_idx=0, target_layer_name=layer_name)
        # print(predictions)
        # Resize heatmap to match the larger image shape (36, 256)
        if heatmap.shape != (36, 256):
            heatmap_resized = zoom(heatmap, (36 / heatmap.shape[0], 256 / heatmap.shape[1]), order=1)
        else:
            heatmap_resized = heatmap

        if idx > filt_num:
            heatmaps_class_0.append(heatmap_resized)

        overlay = overlay_heatmap(heatmap_resized, input_image[0])  # Overlay resized heatmap on the input image
        ax.imshow(input_image[0, :, :, 0], cmap='gray')
        if binary_label == 1 or dont_filter:
            ax.imshow(overlay, cmap='jet', alpha=0.5)
        ax.set_title(f'{layer_name} Heatmap\n(Target Class: 0), {predictions[0].numpy().item():.2f}, {predictions[1].numpy().item():.2f}')
        ax.axis('off')

    # Generate and plot Grad-CAM heatmaps for each layer for target_class_idx=1
    for idx, layer_name in enumerate(layer_names):
        ax = fig.add_subplot(gs[1, idx + 2])
        heatmap, predictions = grad_cam(model, input_image, target_class_idx=1, target_layer_name=layer_name)

        # Resize heatmap to match the larger image shape (36, 256)
        if heatmap.shape != (36, 256):
            heatmap_resized = zoom(heatmap, (36 / heatmap.shape[0], 256 / heatmap.shape[1]), order=1)
        else:
            heatmap_resized = heatmap
        if idx > filt_num:
            heatmaps_class_1.append(heatmap_resized)

        overlay = overlay_heatmap(heatmap_resized, input_image[0])  # Overlay resized heatmap on the input image
        ax.imshow(input_image[0, :, :, 0], cmap='gray')
        if binary_label == 1 or dont_filter:

            ax.imshow(overlay, cmap='jet', alpha=0.5)
        ax.set_title(f'{layer_name} Heatmap\n(Target Class: 1)')
        ax.axis('off')

    # Compute the average of all heatmaps
    avg_0 =  np.mean(heatmaps_class_0 , axis=0)
    avg_1 =  np.mean(heatmaps_class_1 , axis=0)
    avg_heatmap = (avg_0*2 + avg_1)/3


    # Create the subplot for the average heatmap
    ax = fig.add_subplot(gs[0, 6])
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    if binary_label == 1:
        ax.imshow(overlay_heatmap(norm_heatmap(avg_0), input_image[0]), cmap='jet', alpha=0.5)
    ax.set_title('Average Heatmap_0')
    ax.axis('off')


    ax = fig.add_subplot(gs[1, 6])
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    if binary_label == 1:
        ax.imshow(overlay_heatmap(norm_heatmap(avg_1), input_image[0]), cmap='jet', alpha=0.5)
    ax.set_title('Average Heatmap_1')
    ax.axis('off')

    ax = fig.add_subplot(gs[:2, 7:9])
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    if binary_label == 1:
        ax.imshow(overlay_heatmap(norm_heatmap(avg_heatmap), input_image[0]), cmap='jet', alpha=0.5)
    ax.set_title('Average Heatmap_both')
    ax.axis('off')

    plt.tight_layout()


    # Add color gradient for intensity scale
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Colorbar axis (x, y, width, height)
    ColorbarBase(cbar_ax, cmap='jet', norm=Normalize(vmin=0, vmax=1), orientation='horizontal')
    cbar_ax.set_title('Contribution\nIntensity', fontsize=10)

    fig.subplots_adjust(bottom=0.1)

    plt.savefig(f'../../figures_tanh/heatmap_for_layers_{image_idx}.jpg')
    plt.close()



def plot_heatmaps_for_layers_just_one(input_image, model, binary_label, regression_label, image_idx):
    # Define layer names and target class indices
    filt_num = 1

    layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']

    # Create a larger figure for the grid
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(1, 7)

    # Create the image subplot that spans the first 2x2 area
    ax = fig.add_subplot(gs[0, 0])

    # Plot the original image in the top-left (larger area)
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    ax.set_title(f'Original Image\nBinary Label: {binary_label}, Regression Label: {regression_label:.2f}')
    ax.axis('off')

    # Initialize list to store heatmaps for averaging later
    heatmaps_class_0 = []
    heatmaps_class_1 = []

    # Generate and plot Grad-CAM heatmaps for each layer for target_class_idx=0
    for idx, layer_name in enumerate(layer_names):
        ax = fig.add_subplot(gs[0, idx + 1])
        heatmap, predictions = grad_cam(model, input_image, target_layer_name=layer_name)
        # print(predictions)
        # Resize heatmap to match the larger image shape (36, 256)
        if heatmap.shape != (36, 256):
            heatmap_resized = zoom(heatmap, (36 / heatmap.shape[0], 256 / heatmap.shape[1]), order=1)
        else:
            heatmap_resized = heatmap

        if idx > filt_num:
            heatmaps_class_0.append(heatmap_resized)

        overlay = overlay_heatmap(heatmap_resized, input_image[0])  # Overlay resized heatmap on the input image
        ax.imshow(input_image[0, :, :, 0], cmap='gray')
        if binary_label == 1:
            ax.imshow(overlay, cmap='jet', alpha=0.5)
        ax.set_title(f'{layer_name} Heatmap\n(Target Class: 0), {predictions.numpy().item():.2f}')
        ax.axis('off')



    # Compute the average of all heatmaps
    avg_0 =  np.mean(heatmaps_class_0 , axis=0)
    avg_1 =  np.mean(heatmaps_class_1 , axis=0)
    avg_heatmap = (avg_0*2 + avg_1)/3


    # Create the subplot for the average heatmap
    ax = fig.add_subplot(gs[0, 5])
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    if binary_label == 1:
        ax.imshow(overlay_heatmap(norm_heatmap(avg_0), input_image[0]), cmap='jet', alpha=0.5)
    ax.set_title('Average Heatmap_0')
    ax.axis('off')




    ax = fig.add_subplot(gs[0, 6])
    ax.imshow(input_image[0, :, :, 0], cmap='gray')
    if binary_label == 1:
        ax.imshow(overlay_heatmap(norm_heatmap(avg_heatmap), input_image[0]), cmap='jet', alpha=0.5)
    ax.set_title('Average Heatmap_both')
    ax.axis('off')

    plt.tight_layout()


    # Add color gradient for intensity scale
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Colorbar axis (x, y, width, height)
    ColorbarBase(cbar_ax, cmap='jet', norm=Normalize(vmin=0, vmax=1), orientation='horizontal')
    cbar_ax.set_title('Contribution\nIntensity', fontsize=10)

    fig.subplots_adjust(bottom=0.1)

    plt.savefig(f'../../figures_just_relu/heatmap_for_layers_{image_idx}.jpg')
    plt.close()

for image_idx in range(69):
    # Load an input image and preprocess it
    # image_idx = 2  # Choose an image index
    input_image = rxs[image_idx:image_idx + 1]  # Single image batch

    # Extract labels for the current image
    binary_label = labels[image_idx, 0]
    regression_label = labels[image_idx, 1]


    # Example usage:
    plot_heatmaps_for_layers(input_image, model, binary_label, regression_label, f"{image_idx:03}")
    # plot_heatmaps_for_layers_just_one(input_image, model, binary_label, regression_label, f"{image_idx:03}")
    print(image_idx)
    # plot_with_labels_and_explanations(input_image, overlay, binary_label, regression_label)




