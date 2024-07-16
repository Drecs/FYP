from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50))  # Notification type (e.g., 'error')
    icon = db.Column(db.String(10))  # Notification icon (e.g., '&#10008;')
    title = db.Column(db.String(255))  # Notification title
    content = db.Column(db.Text)  # Notification content
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) # Timestamp for notification creation


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime)  # Date and time of the report
    src_ip = db.Column(db.String(15))  # Source IP Address
    dst_ip = db.Column(db.String(15))  # Destination IP Address
    src_port = db.Column(db.Integer)  # Source Port
    dst_port = db.Column(db.Integer)  # Destination Port
    activity_type = db.Column(db.String(50))  # Type of Activity/Anomaly
    description = db.Column(db.Text)  # Description


class NetworkDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ssid = db.Column(db.String(50))
    ip_address = db.Column(db.String(15))
    mac_address = db.Column(db.String(17))
    link_speed = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
