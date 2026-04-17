import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.utils import plot_model
import visualkeras

# Load model Xception
model = Xception(weights=None, input_shape=(224,224,3), classes=2)

# Cara 1: Simpan ke PNG pakai keras
plot_model(model, to_file="xception_model.png", show_shapes=True, show_layer_names=True)

# Cara 2: Visualisasi blok diagram sederhana pakai visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 16)  # biar tulisannya jelas
visualkeras.layered_view(model, legend=True, font=font, to_file='xception_blocks.png')
