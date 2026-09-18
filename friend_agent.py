import requests
import psutil
import socket
import time

SERVER = "http://192.168.120.147:5000/report"

DEVICE_NAME = input(
    "Enter device name (example Laptop-02): "
).strip()

if not DEVICE_NAME:
    DEVICE_NAME = "Friend-Laptop"


def get_local_ip():

    s = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    try:

        s.connect(("8.8.8.8", 80))

        ip = s.getsockname()[0]

    except:

        ip = "Unknown"

    finally:

        s.close()

    return ip


def get_connections():

    connections = []

    try:

        for conn in psutil.net_connections(
            kind="inet"
        ):

            if not conn.raddr:
                continue

            if conn.status not in [
                "ESTABLISHED",
                "CLOSE_WAIT"
            ]:
                continue

            process = "Unknown"

            if conn.pid:

                try:

                    process = psutil.Process(
                        conn.pid
                    ).name()

                except:
                    pass

            connections.append({

                "remote_ip":
                    conn.raddr.ip,

                "port":
                    conn.raddr.port,

                "process":
                    process,

                "status":
                    conn.status

            })

    except Exception as e:

        print("Connection error:", e)

    return connections


print("\n================================")
print("      FRIEND DEVICE AGENT")
print("================================")

print("Device:", DEVICE_NAME)

local_ip = get_local_ip()

print("Local IP:", local_ip)

print("Sending reports to:")
print(SERVER)

print("\nPress CTRL+C to stop.\n")


while True:

    connections = get_connections()

    data = {

        "device":
            DEVICE_NAME,

        "local_ip":
            local_ip,

        "connections":
            connections

    }

    try:

        response = requests.post(
            SERVER,
            json=data,
            timeout=5
        )

        print(
            "Report sent:",
            len(connections),
            "connections"
        )

    except Exception as e:

        print(
            "Cannot connect to DQSA server:",
            e
        )

    time.sleep(5)