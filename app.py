from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import time
from scapy.all import sniff, IP, TCP, UDP

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dqsa_model.pth")

# ============================================================
# DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["application_category"] = df["traffic_type"].str.replace(
    "VPN-", "", regex=False
)

df["encryption_type"] = np.where(
    df["traffic_type"].str.startswith("VPN-"),
    "VPN",
    "Non-VPN"
)

service_mapping = {
    "BROWSING": "Web Browsing",
    "CHAT": "Communication",
    "MAIL": "Communication",
    "STREAMING": "Media Streaming",
    "VOIP": "Real-Time Communication",
    "FT": "File Transfer",
    "P2P": "Peer-to-Peer"
}

df["service_type"] = df["application_category"].map(service_mapping)

app_classes = sorted(df["application_category"].unique())
service_classes = sorted(df["service_type"].unique())
enc_classes = sorted(df["encryption_type"].unique())

X = df.select_dtypes(include=["number"]).values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

print("Applications:", app_classes)
print("Services:", service_classes)
print("Encryption:", enc_classes)

# ============================================================
# DQSA MODEL
# ============================================================

class TaskGatedAttention(nn.Module):

    def __init__(self, d_model=64, heads=4, num_tasks=3):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

        self.task_gates = nn.Parameter(
            torch.zeros(num_tasks, heads)
        )

    def forward(self, x, task_id):

        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(
            batch_size,
            seq_len,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / np.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, v)

        gates = torch.softmax(
            self.task_gates[task_id],
            dim=0
        )

        output = output * gates.view(
            1, self.heads, 1, 1
        )

        output = output.transpose(
            1, 2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.d_model
        )

        return self.output(output)


class SoftQuantization(nn.Module):

    def __init__(self, levels=7):
        super().__init__()

        self.centers = nn.Parameter(
            torch.linspace(-1, 1, levels)
        )

    def forward(self, x):

        bounded = torch.tanh(x)

        distance = (
            bounded.unsqueeze(-1)
            - self.centers
        ) ** 2

        weights = torch.softmax(
            -4.0 * distance,
            dim=-1
        )

        quantized = (
            weights * self.centers
        ).sum(dim=-1)

        return (
            0.8 * bounded +
            0.2 * quantized
        )


class DQSA(nn.Module):

    def __init__(
        self,
        num_features,
        num_app,
        num_service,
        num_enc
    ):
        super().__init__()

        d_model = 64

        self.feature_embedding = nn.Linear(1, d_model)

        self.feature_embedding_id = nn.Parameter(
            torch.randn(
                1,
                num_features,
                d_model
            ) * 0.02
        )

        self.attention = TaskGatedAttention(
            d_model=64,
            heads=4,
            num_tasks=3
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.quantizer = SoftQuantization(levels=7)

        self.app_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_app)
        )

        self.service_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_service)
        )

        self.enc_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_enc)
        )

    def shared_representation(self, x, task_id):

        x = x.unsqueeze(-1)

        x = self.feature_embedding(x)

        x = x + self.feature_embedding_id

        attention_output = self.attention(
            x,
            task_id
        )

        x = self.norm1(
            x + attention_output
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + ffn_output
        )

        x = x.mean(dim=1)

        x = self.quantizer(x)

        return x

    def forward(self, x):

        app_rep = self.shared_representation(x, 0)
        service_rep = self.shared_representation(x, 1)
        enc_rep = self.shared_representation(x, 2)

        return (
            self.app_head(app_rep),
            self.service_head(service_rep),
            self.enc_head(enc_rep)
        )


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = DQSA(
    num_features=X_scaled.shape[1],
    num_app=len(app_classes),
    num_service=len(service_classes),
    num_enc=len(enc_classes)
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("DQSA model loaded successfully!")

# ============================================================
# DEVICE REPORTS
# ============================================================

device_reports = {}

# ============================================================
# OWN LAPTOP LIVE DQSA
# ============================================================

def capture_live_dqsa():

    flows = {}

    # Automatically find this laptop IP
    local_ip = "192.168.120.147"

    def process_packet(packet):

        if IP not in packet:
            return

        src = packet[IP].src

        if src != local_ip:
            return

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
            return

        key = (
            src,
            dst,
            src_port,
            dst_port,
            protocol
        )

        now = time.time()

        if key not in flows:
            flows[key] = {
                "start": now,
                "last": now,
                "packets": 0,
                "bytes": 0,
                "times": [],
                "sizes": []
            }

        flow = flows[key]

        flow["last"] = now
        flow["packets"] += 1
        flow["bytes"] += len(packet)
        flow["times"].append(now)
        flow["sizes"].append(len(packet))

    print("Capturing traffic for 5 seconds...")

    try:
        sniff(
            prn=process_packet,
            store=False,
            timeout=5
        )
    except Exception as e:
        print("Capture error:", e)
        return []

    results = []

    for key, flow in flows.items():

        if flow["packets"] < 5:
            continue

        src, dst, src_port, dst_port, protocol = key

        duration = max(
            flow["last"] - flow["start"],
            0.001
        )

        packets = flow["packets"]
        total_bytes = flow["bytes"]

        flow_pps = packets / duration
        flow_bps = total_bytes / duration

        times = flow["times"]

        intervals = [
            times[i] - times[i - 1]
            for i in range(1, len(times))
        ]

        if intervals:

            min_iat = min(intervals)
            max_iat = max(intervals)
            mean_iat = sum(intervals) / len(intervals)

            std_iat = (
                sum(
                    (x - mean_iat) ** 2
                    for x in intervals
                ) / len(intervals)
            ) ** 0.5

        else:

            min_iat = 0
            max_iat = 0
            mean_iat = 0
            std_iat = 0

        sizes = flow["sizes"]

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

        live_x = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        live_x = scaler.transform(live_x)

        tensor = torch.tensor(
            live_x,
            dtype=torch.float32
        )

        with torch.no_grad():

            app_out, service_out, enc_out = model(tensor)

            app_prob = torch.softmax(
                app_out, dim=1
            )

            service_prob = torch.softmax(
                service_out, dim=1
            )

            enc_prob = torch.softmax(
                enc_out, dim=1
            )

            app_idx = torch.argmax(
                app_prob, dim=1
            ).item()

            service_idx = torch.argmax(
                service_prob, dim=1
            ).item()

            enc_idx = torch.argmax(
                enc_prob, dim=1
            ).item()

        confidence = (
            app_prob[0, app_idx].item()
            + service_prob[0, service_idx].item()
            + enc_prob[0, enc_idx].item()
        ) / 3

        # Process lookup
        process_name = "Unknown"

        try:

            for conn in psutil.net_connections(
                kind="inet"
            ):

                if (
                    conn.laddr
                    and conn.laddr.ip == src
                    and conn.laddr.port == src_port
                ):

                    if conn.pid:

                        try:
                            process_name = psutil.Process(
                                conn.pid
                            ).name()
                        except:
                            pass

                    break

        except:
            pass

        # Human-readable demo application label
        category = app_classes[app_idx]

        if category == "STREAMING":
            app_name = "YouTube"
        elif category == "BROWSING":
            app_name = "Web Browser"
        elif category == "CHAT":
            app_name = "Chat Application"
        elif category == "MAIL":
            app_name = "Email"
        elif category == "VOIP":
            app_name = "Voice / Video Call"
        elif category == "FT":
            app_name = "File Transfer"
        elif category == "P2P":
            app_name = "Peer-to-Peer"
        else:
            app_name = category

        results.append({
            "device": "Laptop-01",
            "local_ip": src,
            "remote_ip": dst,
            "port": dst_port,
            "process": process_name,
            "status": "ESTABLISHED",
            "category": category,
            "app_name": app_name,
            "service": service_classes[service_idx],
            "vpn": enc_classes[enc_idx],
            "confidence": round(
                confidence * 100,
                1
            )
        })

    return results


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    live_predictions = capture_live_dqsa()

    return render_template(
        "index.html",
        live_predictions=live_predictions,
        device_reports=device_reports
    )


# ============================================================
# FRIEND DEVICE REPORT
# ============================================================

@app.route("/report", methods=["POST"])
def report():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No data received"
        }), 400

    device_name = data.get(
        "device",
        "Unknown Device"
    )

    device_reports[device_name] = {
        "local_ip": data.get(
            "local_ip",
            "Unknown"
        ),
        "connections": data.get(
            "connections",
            []
        ),
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    print(
        "Received:",
        device_name
    )

    return jsonify({
        "status": "success"
    })


# ============================================================
# API
# ============================================================

@app.route("/api/devices")
def api_devices():

    return jsonify(device_reports)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       DQSA LIVE MONITOR")
    print("===================================")

    print("\nLocal:")
    print("http://127.0.0.1:5000")

    print("\nLAN:")
    print("http://192.168.120.147:5000")

    print("\nStarting server...\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)