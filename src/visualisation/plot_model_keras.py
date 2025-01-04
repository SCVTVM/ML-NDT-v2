from tensorflow.keras.utils import plot_model
import keras
import visualkeras

model_path = "../training_variants/modelcpnt1c8caf1a-456c-4ce6-aa9f-af4678be622d.keras"
model_path = "../training_variants/modelcpntndt_nn_relu_relu.keras"


model = keras.models.load_model(model_path)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_activations=True)


visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, to_file='output.png') # write to disk
visualkeras.layered_view(model, to_file='output.png', legend=True).show() # write and show

visualkeras.layered_view(model)