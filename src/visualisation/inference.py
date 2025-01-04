
from __future__ import print_function
import sys
import keras

import numpy as np
import matplotlib.pyplot as plt

model_path = "../training_variants/modelcpnt1c8caf1a-456c-4ce6-aa9f-af4678be622d.keras"
data_path  = "../../data/validation/F68B8BC9-C4D5-4848-923E-A68176F821D2.bins"
results_path = "../../data/validation/F68B8BC9-C4D5-4848-923E-A68176F821D2.labels"


def load_labels(labels_file):
    return np.loadtxt(labels_file, dtype=np.float32)

model = keras.models.load_model(model_path)
rxs = np.fromfile(data_path, dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')

predictions = model.predict(rxs)
print(predictions)
labels = load_labels(results_path)
labels = np.array([labels[:, 0]]).T
res = np.abs(predictions-labels)
print(np.hstack((predictions, labels)))

print(res)
print(sum(predictions))
print(sum(labels))
print(sum(res))