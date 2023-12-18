import time
from scapy.all import sniff
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.models import load_model

def monitor_network_traffic(path=r'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/trial/network_traffic_anomaly_model.keras', packet_capture_duration=60, sleep_interval=1):
   start_time = time.time()
   while True:
       # Capture network packets for packet_capture_duration seconds
       end_time = start_time + packet_capture_duration
       network_traffic_data = capture_network_traffic(start_time, end_time) # Implement this function to capture network traffic
       
       # Preprocess the captured network_traffic_data
       processed_data = preprocess_network_traffic(network_traffic_data) # Implement this function
       
       # Load the saved autoencoder model
       anomaly_detector = load_model(path=r'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/trial/network_traffic_anomaly_model.keras')
       
       # Reconstruct the data using the loaded autoencoder model
       reconstructed_data = anomaly_detector.predict(processed_data)
       
       # Calculate the difference between the original data and the reconstructed data
       difference = np.abs(processed_data - reconstructed_data)
       
       # Detect anomalies
       anomalies = difference > threshold
       
       # Print anomalies
       print("Indices of anomalies:", anomalies)
       
       # Wait for the next iteration
       time.sleep(sleep_interval)
       start_time = end_time


def capture_network_traffic(start_time, end_time, interface="eth0"):
   # Sniff network packets on the specified interface within the time window
   packets = sniff(iface=interface, timeout=(end_time - start_time))
   
   # Process packets and extract relevant information
   processed_data = []
   for packet in packets:
       # Extract relevant information from the packet
       packet_data = {
           "src_ip": packet[IP].src,
           "dst_ip": packet[IP].dst,
           "protocol": packet[IP].proto,
           # ... other relevant fields
       }
       processed_data.append(packet_data)
   
   return processed_data


def preprocess_network_traffic(network_traffic_data):
   # Perform preprocessing steps (normalization, feature extraction, etc.)
   processed_features = []
   for packet_data in network_traffic_data:
       # Perform feature extraction and normalization
       feature_vector = [
           packet_data["src_ip"],
           packet_data["dst_ip"],
           packet_data["protocol"],
           # ... other features
       ]
       # Normalize feature values if necessary
       normalized_features = normalize_features(feature_vector)
       processed_features.append(normalized_features)
   
   # Return the preprocessed feature vectors
   return processed_features

def normalize_features(feature_vector):
   # Calculate the minimum and maximum values for each feature
   min_values = [min(column) for column in zip(*feature_vector)]
   max_values = [max(column) for column in zip(*feature_vector)]
   
   # Normalize the feature values
   normalized_features = [(value - min_value) / (max_value - min_value)
                         for min_value, max_value, value in zip(min_values, max_values, feature_vector)]
   
   return normalized_features

