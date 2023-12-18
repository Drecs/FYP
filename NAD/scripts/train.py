import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model





# Load your network traffic data into a Pandas DataFrame (assuming your data is in a CSV file)
data = pd.read_csv('C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/gan_synthetic_data.csv'
)

# Extract features from your dataset (assuming all columns except the target are features)
x_train = data.drop(columns=['service','protocol_type','class','flag'])  # Adjust 'target_column' to your actual target column name

# Load your test network traffic data into a Pandas DataFrame (assuming your data is in a CSV file)
test_data = pd.read_csv('C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/IDS_Data1.csv'
)

# Extract features from your test dataset (similar to how you extracted features from your training data)
x_test = test_data.drop(columns=['service','protocol_type','class','flag']
)  # Adjust 'target_column' to your actual target column name

x_test = x_test.replace('ftp_data', np.nan)  # Replace 'ftp_data' with NaN
x_test = x_test.dropna()  # Optionally, drop rows with NaN values
x_test = np.asarray(x_test).astype('float32')
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

    

    

    


# Assuming you have defined the NetworkTrafficAnomalyDetector class and anomaly_detector object previously
# ...

# Detect anomalies using x_test
#anomalies = anomaly_detector.detect_anomalies(x_test)

class NetworkTrafficAnomalyDetector:
    def __init__(self, input_dim, encoding_dim):
        # Define input layer
        input_layer = Input(shape=(input_dim,))
        
        # Define encoder layers
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)  # The last layer of encoder with 'encoding_dim' units
        
        # Define decoder layers
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)  # The last layer of decoder with 'input_dim' units
        
        # Create autoencoder model
        self.autoencoder_model = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder_model.summary()  # Print model summary

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        # Compile the autoencoder model
        self.autoencoder_model.compile(optimizer=optimizer, loss=loss)
    
    def train(self, x_train, epochs=100, batch_size=32):
        # Train the autoencoder model
        self.autoencoder_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    def detect_anomalies(self, x_data, threshold=0.01):
        # Detect anomalies using reconstruction error
        decoded_data = self.autoencoder_model.predict(x_data)
        mse = np.mean(np.square(x_data - decoded_data), axis=1)
        anomalies = np.where(mse > threshold)
        return anomalies

# Example usage:
# Assuming x_train is your network traffic training data (NumPy array)
# Assuming x_test is your network traffic test data (NumPy array)
input_dim = x_train.shape[1]  # Input dimension based on your network traffic features
encoding_dim = 32  # Example encoding dimension



# Create an instance of the NetworkTrafficAnomalyDetector class
anomaly_detector = NetworkTrafficAnomalyDetector(input_dim, encoding_dim)

# Compile and train the autoencoder model using x_train
anomaly_detector.compile()
anomaly_detector.train(x_train, epochs=100, batch_size=32)

# After training the model
anomaly_detector.autoencoder_model.save('network_traffic_anomaly_model.keras')
print("Model saved successfully.")

#loaded_model = load_model('network_traffic_anomaly_model.keras')

# Assuming you have new test data in the variable new_x_test
#new_anomalies = loaded_model.predict(new_x_test)  # Use the loaded model for predictions


# Detect anomalies using x_test
anomalies = anomaly_detector.detect_anomalies(x_test)

# Print the indices of anomalous samples
print("Indices of anomalies:", anomalies)


