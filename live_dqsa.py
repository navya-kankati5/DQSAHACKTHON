import os
import time
import numpy as np
import pandas as pd
import torch
from scapy.all import sniff, IP, TCP, UDP
from sklearn.preprocessing import StandardScaler

from app import DQSA


# =========================
# SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dqsa_model.pth")

LOCAL_IP = "192.168.120.147"
CAPTURE_TIME = 15


# =========================
# LABELS
# =========================
APP_CLASSES = [
    "BROWSING",
    "CHAT",
    "FT",
    "MAIL",
    "P2P",
    "STREAMING",
    "VOIP"
]

SERVICE_CLASSES = [
    "Communication",
    "File Transfer",
    "Media Streaming",
    "Peer-to-Peer",
    "Real-Time Communication",
    "Web Browsing"
]

ENC_CLASSES = [
    "Non-VPN",
    "VPN"
]


# =========================
# LOAD DATA + SCALER
# =========================
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.select_dtypes(include=["number"]).values

scaler = StandardScaler()
scaler.fit(X)

print("Dataset loaded:", len(df), "rows")


# =========================
# LOAD DQSA MODEL
# =========================
print("\nLoading DQSA model...")

model = DQSA(
    num_features=23,
    num_app=7,
    num_service=6,
    num_enc=2
)

state = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()

print("DQSA model loaded successfully!")


# =========================
# LIVE FLOW STORAGE
# =========================
flows = {}


def get_flow_key(packet):

    if IP not in packet:
        return None

    src = packet[IP].src
    dst = packet[IP].dst

    if TCP in packet:
        protocol = "TCP"
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport

    elif UDP in packet:
        protocol = "UDP"
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    else:
        return None

    return (
        src,
        dst,
        src_port,
        dst_port,
        protocol
    )


def process_packet(packet):

    key = get_flow_key(packet)

    if key is None:
        return

    src, dst, src_port, dst_port, protocol = key

    # Only monitor our own laptop
    if src != LOCAL_IP:
        return

    now = time.time()

    if key not in flows:

        flows[key] = {
            "start": now,
            "last": now,
            "packets": 0,
            "bytes": 0,
            "packet_times": [],
            "packet_sizes": []
        }

    flow = flows[key]

    flow["last"] = now
    flow["packets"] += 1
    flow["bytes"] += len(packet)

    flow["packet_times"].append(now)
    flow["packet_sizes"].append(len(packet))


# =========================
# FEATURE EXTRACTION
# =========================
def get_features(key):

    flow = flows[key]

    duration = max(
        flow["last"] - flow["start"],
        0.001
    )

    packets = flow["packets"]
    total_bytes = flow["bytes"]

    flow_pps = packets / duration
    flow_bps = total_bytes / duration

    times = flow["packet_times"]

    if len(times) > 1:

        intervals = [
            times[i] - times[i - 1]
            for i in range(1, len(times))
        ]

        min_iat = min(intervals)
        max_iat = max(intervals)
        mean_iat = sum(intervals) / len(intervals)

        if len(intervals) > 1:

            mean = mean_iat

            std_iat = (
                sum(
                    (x - mean) ** 2
                    for x in intervals
                )
                / len(intervals)
            ) ** 0.5

        else:
            std_iat = 0.0

    else:

        min_iat = 0.0
        max_iat = 0.0
        mean_iat = 0.0
        std_iat = 0.0

    sizes = flow["packet_sizes"]

    min_size = min(sizes)
    max_size = max(sizes)
    mean_size = sum(sizes) / len(sizes)

    features = [

        duration,
        total_bytes,
        total_bytes,

        min_size,
        min_size,

        max_size,
        max_size,

        mean_size,
        mean_size,

        flow_pps,
        flow_bps,

        min_iat,
        max_iat,
        mean_iat,
        std_iat,

        0.0,
        0.0,
        duration,
        0.0,

        0.0,
        0.0,
        0.0,
        0.0
    ]

    return features


# =========================
# CAPTURE
# =========================
print("\n======================================")
print("        LIVE DQSA MONITOR")
print("======================================")

print("\nStarting packet capture for 15 seconds...")
print("Open Chrome and visit Google or YouTube.\n")

sniff(
    prn=process_packet,
    store=False,
    timeout=CAPTURE_TIME
)

print("\nCapture finished.")


# =========================
# DQSA PREDICTION
# =========================
print("\n======================================")
print("        LIVE DQSA PREDICTIONS")
print("======================================\n")

found = False

for key in flows:

    src, dst, src_port, dst_port, protocol = key

    flow = flows[key]

    # Ignore tiny flows
    if flow["packets"] < 3:
        continue

    found = True

    features = get_features(key)

    X_live = np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    X_live = scaler.transform(X_live)

    tensor = torch.tensor(
        X_live,
        dtype=torch.float32
    )

    with torch.no_grad():

        app_out, service_out, enc_out = model(tensor)

        app_prob = torch.softmax(
            app_out,
            dim=1
        )

        service_prob = torch.softmax(
            service_out,
            dim=1
        )

        enc_prob = torch.softmax(
            enc_out,
            dim=1
        )

        app_idx = torch.argmax(
            app_prob,
            dim=1
        ).item()

        service_idx = torch.argmax(
            service_prob,
            dim=1
        ).item()

        enc_idx = torch.argmax(
            enc_prob,
            dim=1
        ).item()

        confidence = (
            app_prob[0, app_idx].item()
            + service_prob[0, service_idx].item()
            + enc_prob[0, enc_idx].item()
        ) / 3

    print("--------------------------------------")

    print(
        f"Device      : Laptop-01"
    )

    print(
        f"Local IP    : {src}"
    )

    print(
        f"Remote IP   : {dst}"
    )

    print(
        f"Port        : {dst_port}"
    )

    print(
        f"Protocol    : {protocol}"
    )

    print(
        f"Packets     : {flow['packets']}"
    )

    print(
        f"Application : {APP_CLASSES[app_idx]}"
    )

    print(
        f"Service     : {SERVICE_CLASSES[service_idx]}"
    )

    print(
        f"VPN         : {ENC_CLASSES[enc_idx]}"
    )

    print(
        f"Confidence  : {confidence * 100:.1f}%"
    )


if not found:

    print(
        "No sufficiently large outgoing flows captured."
    )

print("\nDone.")