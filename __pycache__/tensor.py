import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Create a Sequential model
model = Sequential([
    Dense(128, input_shape=(10,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Visualize the model architecture
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# Show the image
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
