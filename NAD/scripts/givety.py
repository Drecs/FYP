import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from scripts.sniff import capture_packets, extract_features
from scapy.layers.inet import IP  # Import the IP layer from Scapy

# Load the trained model
model = load_model(
    'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/classifier_model.h5')

# Function to extract additional features from packets and return them as a DataFrame


def extract_additional_features(packets):
    # Placeholder for additional features (protocol_type, service, flag)
    additional_features = []

    # Extract and add additional features to the DataFrame
    for packet in packets:
        # Extract additional features from the IP layer
        if IP in packet:
            protocol_type = packet[IP].proto
        else:
            protocol_type = 0  # Default value if protocol type is not available

        # Placeholder for 'service' and 'flag'
        service = 0
        flag = 0

        # Append the extracted features as a list
        additional_features.append([protocol_type, service, flag])

    return pd.DataFrame(additional_features, columns=['protocol_type', 'service', 'flag'])


def preprocess_data(data):
    # Standardize numerical features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data

# Function to check for attack types in network traffic


def check_attack(packets):
    # Initialize collected_traffic list
    collected_traffic = []

    # Extract features from packets
    for packet in packets:
        # Extract features from the packet
        features = extract_features(packet)
        if len(features) != 12:
            # If the number of features is not 12, skip this packet
            continue
        collected_traffic.append(features)

    # Load and preprocess the collected network traffic data
    df = pd.DataFrame(collected_traffic, columns=['length', 'src', 'dst', 'sport', 'dport',
                      'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12'])

    # Extract additional features
    additional_features_df = extract_additional_features(packets)

    # Concatenate the additional features DataFrame with the original data
    data = pd.concat([df, additional_features_df], axis=1)

    # Call the function to preprocess data
    processed_data = preprocess_data(data)

    # Ensure that the processed_data has shape (None, 41)
    if processed_data.shape[1] < 41:
        processed_data = np.pad(
            processed_data, ((0, 0), (0, 41 - processed_data.shape[1])), mode='constant')

    return processed_data


# Capture network packets
interface = 'Wi-Fi'  # Replace with your interface name
packet_count = 100  # Number of packets to capture
packets = capture_packets(interface, packet_count)

# Call the function to check for attacks
processed_packets = check_attack(packets)

# Make predictions
y_pred_probabilities = model.predict(processed_packets)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Reverse encoding for target variable
y_encoder = LabelEncoder()
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'y_encoder_classes.npy')
y_encoder.classes_ = np.load(file_path, allow_pickle=True)
y_pred_labels = y_encoder.inverse_transform(y_pred)

# Check for specified attack types
attack_types = ["r2l", "normal", "probe", "dos"]
detected_attacks = []
for attack_type in attack_types:
    if attack_type in y_pred_labels:
        detected_attacks.append(attack_type)

# Custom messages for each attack type
attack_messages = {
    "normal": "Disturbance due to slow internet connection or weak signal.",
    "probe": "There seems to be unusual packet activity in the network. Check the systems for a possible probe attack.",
    "dos": "There seems to be a sudden influx of traffic in the network. Check the systems for a possible DoS attack.",
    "r2l": "There seems to be unusual packet activity, especially with the IPs. Please check the systems for a possible R2L attack."
}

# Print custom messages for detected attacks
if detected_attacks:
    for attack_type in detected_attacks:
        print(attack_messages[attack_type])
else:
    print("There seems to be a delay in the transmission of data packets between devices in the network, which may be caused by network congestion, distance, or processing delays.")
