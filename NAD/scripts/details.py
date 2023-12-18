import platform
import socket
import netifaces
import psutil
import subprocess
import uuid
from datetime import datetime
import pytz
from scripts.models import db
from flask_sqlalchemy import SQLAlchemy
from scripts.models import NetworkDetails


def get_network_details():
    # Initialize an empty dictionary to store the network details
    network_details = {}

    # Get the SSID
    if platform.system() == "Windows":
        ssid = subprocess.check_output(["netsh", "wlan", "show", "interfaces"]).decode(
            'utf-8').split("\n")[5].split(":")[1].strip()
    elif platform.system() == "Linux":
        ssid = subprocess.check_output(
            ["iwgetid", "-r"]).decode('utf-8').strip()
    else:
        ssid = "Unknown"
    network_details['ssid'] = ssid

    # Get the IPv4 address
    ip_address = socket.gethostbyname(socket.gethostname())
    network_details['ip_address'] = ip_address

    # Get the MAC address
    mac_address = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff)
                            for i in range(0, 8*6, 8)][::-1])
    network_details['mac_address'] = mac_address

    # Get the network interface
    network_interface = None
    for interface in netifaces.interfaces():
        if netifaces.AF_INET in netifaces.ifaddresses(interface):
            network_interface = interface
            break

    if network_interface:
        # Get the link speed
        net_io = psutil.net_io_counters(pernic=True)
        if network_interface in net_io:
            link_speed = net_io[network_interface].bytes_sent + \
                net_io[network_interface].bytes_recv
            network_details['link_speed'] = link_speed
        else:
            print("Network interface not found in net_io")
    else:
        print("No active network interface found.")

     # Save the network details to the database
    save_network_details_to_db(network_details)

    # Return the dictionary
    return network_details


def save_network_details_to_db(network_details):
    details = NetworkDetails(
        ssid=network_details['ssid'],
        ip_address=network_details['ip_address'],
        mac_address=network_details['mac_address'],
        link_speed=network_details.get('link_speed', None)
    )
    db.session.add(details)
    db.session.commit()


def get_recent_network_details(limit=5):
    # Define the target timezone
    target_timezone = pytz.timezone('Etc/GMT-2')

    # Fetch the most recent 5 network details from the database
    recent_details = NetworkDetails.query.order_by(
        NetworkDetails.timestamp.desc()).limit(limit).all()

    # Return a list of dictionaries containing network details with timestamp in the desired timezone
    return [{
        'ssid': detail.ssid,
        'ip_address': detail.ip_address,
        'mac_address': detail.mac_address,
        'link_speed': detail.link_speed,
        'timestamp': detail.timestamp.astimezone(target_timezone).strftime('%Y-%m-%d %H:%M:%S')
    } for detail in recent_details]
