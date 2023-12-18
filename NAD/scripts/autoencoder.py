# Import required libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


class Encoder:
    def __init__(self, input_dim, encoding_dim):
        # Define input layer
        input_layer = Input(shape=(input_dim,))

        # Define encoder layers
        encoded = Dense(encoding_dim, activation='relu')(input_layer)

        # Create encoder model
        self.encoder_model = Model(inputs=input_layer, outputs=encoded)
        self.encoder_model.summary()  # Print model summary

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        # Compile the encoder model
        self.encoder_model.compile(optimizer=optimizer, loss=loss)

    def train(self, x_train, epochs=100, batch_size=32):
        # Train the encoder model
        self.encoder_model.fit(
            x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    def encode(self, x):
        # Encode input data
        encoded_data = self.encoder_model.predict(x)
        return encoded_data


# Example usage:
# Create an instance of the Encoder class with input dimension and encoding dimension
input_dim = 784  # Example input dimension (for MNIST data)
encoding_dim = 32  # Example encoding dimension
encoder = Encoder(input_dim, encoding_dim)

# Compile the encoder model
encoder.compile()

# Assuming x_train is your training data (NumPy array)
# Train the encoder model
# encoder.train(x_train, epochs=100, batch_size=32)

# Example encoding usage:
# encoded_data = encoder.encode(x_test)  # Encode x_test data
