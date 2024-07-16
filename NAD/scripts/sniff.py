from scapy.all import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk
from datetime import datetime


# Load your trained Keras autoencoder model
autoencoder_model = load_model(
    'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/network_traffic_anomaly_model.keras')


# Define a function to capture network packets
def capture_packets(interface, count):
    packets = sniff(iface=interface, count=count)
    for packet in packets:
        packet.show()  # Print the packet
        print(f"Packet type: {packet.__class__.name}")  # Print the packet type
    return packets


# Define a function to extract features from the captured packets
def extract_features(packets):
    features = []
    for packet in packets:
        try:
            # Convert MAC addresses to integer values
            src = int(packet.src.replace(':', ''), 16)
            dst = int(packet.dst.replace(':', ''), 16)
            # Extract 12 features from the packet
            feature = [len(packet), src, dst, packet.sport,
                       packet.dport, 0, 0, 0, 0, 0, 0, 0]
            features.append(feature)
        except AttributeError:
            print(
                f"One or more fields not found in packet of type {packet.__class__.name}")
    # Reshape the data to have 12 columns
    data = np.array(features).reshape(-1, 12)
    return data


# Define a function to detect anomalies using the autoencoder model
def detect_anomalies(data):
    # Pass the data through the autoencoder to get the reconstructed data
    reconstructed_data = autoencoder_model.predict(data)

    # Calculate the reconstruction error
    reconstruction_error = np.mean(
        np.power(data - reconstructed_data, 2), axis=1)

    # Detect anomalies based on the reconstruction error
    # Example threshold calculation
    threshold = np.mean(reconstruction_error) + 2 * \
        np.std(reconstruction_error)
    anomalies = reconstruction_error > threshold
    return anomalies, reconstruction_error


# Capture network packets
packets = capture_packets('Wi-Fi', 100)

# Extract features from the captured packets
data = extract_features(packets)

# Detect anomalies in the network traffic
anomalies = detect_anomalies(data)

# Print the indices of anomalous samples
print("Indices of anomalies:", anomalies)


def generate_anomalies_plot(anomalies, reconstruction_errors):
    indices = np.where(anomalies)[0]

    # Separate anomalies and their indices
    true_anomalies = reconstruction_errors[indices]
    false_anomalies = reconstruction_errors[~anomalies]
    false_indices = np.where(~anomalies)[0]

    # Plot true anomalies in red above the zero line
    fig, ax = plt.subplots()
    ax.scatter(indices, true_anomalies, c='red',
               label='True Anomalies', marker='o')

    # Plot false anomalies in blue on the zero line
    ax.scatter(false_indices, [0] * len(false_indices),
               c='blue', label='False Anomalies', marker='o')

    ax.set_title('Anomalies Detection')
    ax.set_xlabel('Packets')
    ax.set_ylabel('Anomaly magnitude')

    # Save the plot as an image file
    plot_filename = 'anomalies_plot.png'

    # Create a new Tkinter Tk object
    root = Tk()

    # Use FigureCanvasTkAgg with the Tk object as the master
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # Save the plot in the 'static' folder to serve it statically
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    # Uncomment the next line if you want to save the plot as an image file
    canvas.print_png(f'static/{plot_filename}')

    # Explicitly close the Tkinter window
    root.destroy()

    return plot_filename


def generate_report(anomalies, detected_attacks, attack_messages):
    indices = np.where(anomalies)[0]
    reports = []
    for idx in indices:
        # Convert attack types to strings as needed
        attack_type = str(detected_attacks[idx])

        # Determine the description based on the detected attack type
        if attack_type in attack_messages:
            description = attack_messages[attack_type]
        else:
            description = "There seems to be be a mere disturbance in the network due to slow internet connection, which may lead to issues like packet loss"  # Default message

        # Extract relevant information from the anomalous packet
        date_time = datetime.now()
        src_ip = packets[idx].src
        dst_ip = packets[idx].dst
        src_port = packets[idx].sport
        dst_port = packets[idx].dport
        activity_type = "Anomaly"  # You can customize this based on your needs

        # Append the report to the list
        report = {
            "date_time": date_time,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "activity_type": activity_type,
            "description": description,
        }
        reports.append(report)

    return reports


def generate_progress_plot(anomalies):
    total_packets = len(anomalies)
    if total_packets == 0:
        # If there are no anomalies, set the percentage to 0
        percentage = 0
    else:
        # Calculate the percentage of anomalous packets
        percentage = np.sum(anomalies) / total_packets * 100

    # Create a circular progress plot
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    wedges, texts, autotexts = ax.pie(
        [percentage, 100 - percentage],
        labels=[f'Anomalies: {percentage:.2f}%', 'Normal'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['red', 'lightgray']
    )

    # Customize the plot
    plt.setp(autotexts, size=12, color='white')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Save the plot as an image file
    progress_plot_filename = 'progress_plot.png'
    plt.savefig(f'static/{progress_plot_filename}')

    return progress_plot_filename
