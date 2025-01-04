import keras
import netron

# Load your Keras model
model_path = "../training_variants/modelcpnt1c8caf1a-456c-4ce6-aa9f-af4678be622d.keras"
model_path = "../training_variants/modelcpntndt_nn_relu_relu.keras"

model = keras.models.load_model(model_path)

# Start Netron server
netron.start(model_path)
