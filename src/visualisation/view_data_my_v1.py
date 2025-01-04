import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

import re
import json

import matplotlib as mpl
mpl.use("TkAgg")

# Define folder path
path = "../../data/training/"

# Get a list of .bins, .jsons, and .labels files
bins_files = sorted([f for f in listdir(path) if f.endswith('.bins')])
jsons_files = sorted([f for f in listdir(path) if f.endswith('.jsons')])
labels_files = sorted([f for f in listdir(path) if f.endswith('.labels')])

# Select the first set of files for demonstration
bins_file = join(path, bins_files[0])
jsons_file = join(path, jsons_files[0])
labels_file = join(path, labels_files[0])


# Load images from the .bins file
def load_images(file_path):
    img_data = np.fromfile(file_path, dtype=np.uint16).astype('float32')
    img_data -= img_data.mean()  # normalize
    img_data /= img_data.std() + 1e-5  # avoid division by zero
    img_data = img_data.reshape((-1, 256, 256))  # 100 images of 256x256
    return img_data


# Function to load flaw metadata from a concatenated JSONs file
def load_metadata(json_file):
    with open(json_file, 'r') as f:
        # Read the entire line and split each JSON object
        data = f.read()
        # Use regex to split on '}{' and keep braces with each entry
        entries = re.split(r'}\s*{', data)
        # Add braces to each entry as needed
        entries = ['{' + e if not e.startswith('{') else e for e in entries]
        entries = [e + '}' if not e.endswith('}') else e for e in entries]
    # Parse each entry
    print(entries[0])
    metadata = [json.loads(entry) for entry in entries]
    return metadata


# Load labels from the .labels file
def load_labels(labels_file):
    return np.loadtxt(labels_file, dtype=np.float32)


# Load data
images = load_images(bins_file)
metadata = load_metadata(jsons_file)
labels = load_labels(labels_file)

# Display a few images with annotations
num_images_to_show = 10
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')

    # Annotate with flaw information if it exists
    if labels[i, 0] == 1:  # Flaw exists
        flaw_info = f"Flaw Size: {labels[i, 1]}"
        plt.title(flaw_info, fontsize=8)

plt.show()