from scripts.sniff import capture_packets, extract_features, detect_anomalies, generate_anomalies_plot, generate_report, generate_progress_plot
from scripts.details import get_network_details, get_recent_network_details, save_network_details_to_db
from flask import Flask, render_template
from flask import flash
from flask import session
from scapy.all import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scripts.models import db, Notification, Report, NetworkDetails
from datetime import datetime
import pytz
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import sys
sys.path.append('C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts')


app = Flask(
    __name__, template_folder='C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/templates')


# Set the secret key
app.secret_key = 'progressinsession'


# Adjust the database URI as needed
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db.init_app(app)
migrate = Migrate(app, db)


# Load your trained Keras autoencoder model
autoencoder_model = load_model(
    'C:/Users/Nec/Desktop/EXTRA CURR/zips/NAD/scripts/network_traffic_anomaly_model.keras')

# Import functions from your script

# Function to save notifications to the database


def save_notification(notification_type, notification_icon, notification_title, notification_content, timestamp=None):
    # If timestamp is not provided, set it to the current UTC time
    if timestamp is None:
        timestamp = datetime.utcnow()

    notification = Notification(
        type=notification_type,
        icon=notification_icon,
        title=notification_title,
        content=notification_content,
        timestamp=timestamp
    )

    # Return the created notification object
    return notification


@app.route('/')
def index():
    # Initialize notification to None
    notification = None

    # Capture network packets
    packets = capture_packets('Wi-Fi', 100)

    # Extract features from the captured packets
    data = extract_features(packets)

    # Detect anomalies in the network traffic
    anomalies, reconstruction_errors = detect_anomalies(data)

    # Print the indices of anomalous samples
    print("Indices of anomalies:", np.where(anomalies)[0])

    # Generate and save the anomalies plot
    plot_filename = generate_anomalies_plot(anomalies, reconstruction_errors)

    # Generate and save the circular progress plot
    progress_plot_filename = generate_progress_plot(anomalies)

    # Check if anomalies are detected and save notification to the database
    if any(anomalies):
        # Set the time zone to UTC
        utc_timezone = pytz.timezone('Etc/GMT-2')

        # Get the current time in UTC
        current_time = datetime.now(utc_timezone)

        # Save notification and get the created notification object
        notification = save_notification(
            notification_type='error',
            notification_icon='&#10008;',
            notification_title='Network Anomaly Detected',
            notification_content='The system has detected unusual activity on the network. The system has identified a potential security breach or irregular data traffic pattern. Immediate investigation and action are highly recommended.',
            timestamp=current_time
        )

        # Add the notification to the session for immediate display
        session['notification'] = {
            'type': 'error',
            'icon': '&#10008;',
            'title': 'Network Anomaly Detected',
            'content': 'The system has detected unusual activity on the network. The system has identified a potential security breach or irregular data traffic pattern. Immediate investigation and action are highly recommended.'
        }

        # Add the notification to the database session
        db.session.add(notification)
        db.session.commit()

        # Retrieve the notification from the session
        notification = session.pop('notification', None)

        # Generate reports for anomalies
        reports_data = generate_report(anomalies)

        # Save reports to the database
        for report_data in reports_data:
            report = Report(**report_data)
            db.session.add(report)
            db.session.commit()

    return render_template('home.html', plot_filename=plot_filename, progress_plot_filename=progress_plot_filename, notification=notification)


@app.route('/network_details')
def network_details():
    # Call your Python script function to get current network details
    current_network_details = get_network_details()

    # Save the current network details to the database
    save_network_details_to_db(current_network_details)

    # Fetch the most recent 5 network details from the database
    recent_network_details = get_recent_network_details(limit=5)

    # Pass the current and the most recent 5 network details to the template
    return render_template('netdet.html', current_network_details=current_network_details, recent_network_details=recent_network_details)


@app.route('/notifications')
def notifications():
    # Fetch notifications from the database, ordered by timestamp in descending order
    notifications = Notification.query.order_by(
        Notification.timestamp.desc()).all()
    return render_template('notifications.html', notifications=notifications)


@app.route('/reports')
def reports():
    # Fetch all reports from the database, ordered by date_time in descending order
    reports = Report.query.order_by(Report.date_time.desc()).all()
    return render_template('reports.html', reports=reports)


if __name__ == '__main__':
    app.run(debug=True)
