from flask import Flask, render_template, request, jsonify
import psutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import os


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "traffic_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "dqsa_model.pth"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CREATE LABELS
# ============================================================

df["application_category"] = df["traffic_type"].str.replace(
    "VPN-", "", regex=False
)

df["encryption_type"] = df["traffic_type"].apply(
    lambda x: "VPN" if x.startswith("VPN-") else "Non-VPN"
)


service_map = {
    "BROWSING": "Web Browsing",
    "CHAT": "Communication",
    "MAIL": "Communication",
    "STREAMING": "Media Streaming",
    "VOIP": "Real-Time Communication",
    "FT": "File Transfer",
    "P2P": "Peer-to-Peer"
}


df["service_type"] = df["application_category"].map(
    service_map
)


# ============================================================
# FEATURES
# ============================================================

X = df.select_dtypes(
    include=["number"]
).values.astype(np.float32)


app_classes = sorted(
    df["application_category"].unique()
)

service_classes = sorted(
    df["service_type"].unique()
)

enc_classes = sorted(
    df["encryption_type"].unique()
)


print("Applications:", app_classes)
print("Services:", service_classes)
print("Encryption:", enc_classes)


# ============================================================
# STANDARD SCALER
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X).astype(
    np.float32
)


# ============================================================
# EXACT DQSA ARCHITECTURE
# ============================================================

class TaskGatedAttention(nn.Module):

    def __init__(
        self,
        d_model=64,
        heads=4,
        num_tasks=3
    ):

        super().__init__()

        self.heads = heads

        self.head_dim = d_model // heads

        self.q = nn.Linear(
            d_model,
            d_model
        )

        self.k = nn.Linear(
            d_model,
            d_model
        )

        self.v = nn.Linear(
            d_model,
            d_model
        )

        self.output = nn.Linear(
            d_model,
            d_model
        )

        self.task_gates = nn.Parameter(
            torch.zeros(
                num_tasks,
                heads
            )
        )


    def forward(
        self,
        x,
        task_id
    ):

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
        )


        scores = scores / np.sqrt(
            self.head_dim
        )


        attention = torch.softmax(
            scores,
            dim=-1
        )


        head_output = torch.matmul(
            attention,
            v
        )


        gates = torch.softmax(
            self.task_gates[task_id],
            dim=0
        )


        gates = gates.view(
            1,
            self.heads,
            1,
            1
        )


        head_output = (
            head_output * gates
        )


        head_output = head_output.transpose(
            1,
            2
        ).contiguous()


        head_output = head_output.view(
            batch_size,
            seq_len,
            self.heads * self.head_dim
        )


        return self.output(
            head_output
        )


# ============================================================
# SOFT QUANTIZATION
# ============================================================

class SoftQuantization(nn.Module):

    def __init__(self, levels=7):

        super().__init__()

        self.levels = levels

        self.centers = nn.Parameter(
            torch.linspace(
                -1.0,
                1.0,
                levels
            )
        )


    def forward(self, x):

        bounded = torch.tanh(x)


        distance = (
            bounded.unsqueeze(-1)
            -
            self.centers
        ) ** 2


        weights = torch.softmax(
            -4.0 * distance,
            dim=-1
        )


        quantized = (
            weights * self.centers
        ).sum(dim=-1)


        return (
            0.8 * bounded
            +
            0.2 * quantized
        )


# ============================================================
# DQSA
# ============================================================

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


        self.feature_embedding = nn.Linear(
            1,
            d_model
        )


        self.feature_embedding_id = nn.Parameter(
            torch.randn(
                1,
                num_features,
                d_model
            ) * 0.02
        )


        self.attention = TaskGatedAttention(
            d_model=d_model,
            heads=4,
            num_tasks=3
        )


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                128,
                d_model
            )
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.quantizer = SoftQuantization(
            levels=7
        )


        self.app_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_app
            )
        )


        self.service_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_service
            )
        )


        self.enc_head = nn.Sequential(

            nn.Linear(
                d_model,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                num_enc
            )
        )


    def shared_representation(
        self,
        x,
        task_id
    ):

        x = x.unsqueeze(-1)


        x = self.feature_embedding(x)


        x = (
            x
            +
            self.feature_embedding_id
        )


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

        app_rep = self.shared_representation(
            x,
            0
        )


        service_rep = self.shared_representation(
            x,
            1
        )


        enc_rep = self.shared_representation(
            x,
            2
        )


        app_output = self.app_head(
            app_rep
        )


        service_output = self.service_head(
            service_rep
        )


        enc_output = self.enc_head(
            enc_rep
        )


        return (
            app_output,
            service_output,
            enc_output
        )


# ============================================================
# LOAD TRAINED DQSA
# ============================================================

device = torch.device("cpu")


model = DQSA(
    num_features=X_scaled.shape[1],
    num_app=len(app_classes),
    num_service=len(service_classes),
    num_enc=len(enc_classes)
)


print("\nLoading trained model...")


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.to(device)

model.eval()


print("DQSA model loaded successfully!")


# ============================================================
# MULTI DEVICE STORAGE
# ============================================================

device_reports = {}


# ============================================================
# LOCAL CONNECTIONS
# ============================================================

def get_live_connections():

    connections = []


    try:

        network_connections = psutil.net_connections(
            kind="inet"
        )

    except Exception as e:

        print("Connection error:", e)

        return connections


    for conn in network_connections:

        if not conn.raddr:
            continue


        if conn.status not in [
            "ESTABLISHED",
            "CLOSE_WAIT"
        ]:
            continue


        try:
            local_ip = conn.laddr.ip
        except:
            local_ip = "Unknown"


        try:
            remote_ip = conn.raddr.ip
        except:
            remote_ip = "Unknown"


        try:
            port = conn.raddr.port
        except:
            port = 0


        process_name = "Unknown"


        if conn.pid:

            try:

                process_name = psutil.Process(
                    conn.pid
                ).name()

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied
            ):

                process_name = "Unknown"


        # Simple demo monitoring flag
        common_ports = [
            53,
            80,
            443,
            5222,
            5228
        ]


        if port in common_ports:

            status = "NORMAL"

        else:

            status = "SUSPICIOUS"


        connections.append({

            "local_ip":
                local_ip,

            "remote_ip":
                remote_ip,

            "port":
                port,

            "process":
                process_name,

            "status":
                conn.status,

            "monitoring_status":
                status

        })


    return connections


# ============================================================
# RECEIVE OTHER AUTHORIZED DEVICES
# ============================================================

@app.route(
    "/report",
    methods=["POST"]
)
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


    connections = data.get(
        "connections",
        []
    )


    device_reports[device_name] = {

        "device":
            device_name,

        "connections":
            connections,

        "timestamp":
            time.time()

    }


    print(
        f"Received {len(connections)} "
        f"connections from {device_name}"
    )


    return jsonify({

        "message":
            "Data received",

        "device":
            device_name

    })


# ============================================================
# DEVICE API
# ============================================================

@app.route("/api/devices")
def api_devices():

    return jsonify(
        device_reports
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    local_connections = (
        get_live_connections()
    )


    return render_template(
        "index.html",
        connections=local_connections,
        device_reports=device_reports
    )


# ============================================================
# DQSA PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # IMPORTANT:
    # This is currently a dataset-sample prediction.
    # Live DQSA will be connected after the flow
    # feature extraction stage.


    sample = torch.tensor(
        X_scaled[0:1],
        dtype=torch.float32
    ).to(device)


    with torch.no_grad():

        app_out, service_out, enc_out = model(
            sample
        )


    # --------------------------------------------------------
    # APPLICATION
    # --------------------------------------------------------

    app_prob = torch.softmax(
        app_out,
        dim=1
    )


    app_index = torch.argmax(
        app_prob,
        dim=1
    ).item()


    application = app_classes[
        app_index
    ]


    app_confidence = (
        app_prob[
            0,
            app_index
        ].item()
        * 100
    )


    # --------------------------------------------------------
    # SERVICE
    # --------------------------------------------------------

    service_prob = torch.softmax(
        service_out,
        dim=1
    )


    service_index = torch.argmax(
        service_prob,
        dim=1
    ).item()


    service = service_classes[
        service_index
    ]


    service_confidence = (
        service_prob[
            0,
            service_index
        ].item()
        * 100
    )


    # --------------------------------------------------------
    # VPN
    # --------------------------------------------------------

    enc_prob = torch.softmax(
        enc_out,
        dim=1
    )


    enc_index = torch.argmax(
        enc_prob,
        dim=1
    ).item()


    encryption = enc_classes[
        enc_index
    ]


    enc_confidence = (
        enc_prob[
            0,
            enc_index
        ].item()
        * 100
    )


    return render_template(

        "index.html",

        connections=
            get_live_connections(),

        device_reports=
            device_reports,

        application=
            application,

        app_confidence=
            round(
                app_confidence,
                2
            ),

        service=
            service,

        service_confidence=
            round(
                service_confidence,
                2
            ),

        encryption=
            encryption,

        enc_confidence=
            round(
                enc_confidence,
                2
            )

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )

    print(
        "          DQSA NETWORK MONITOR"
    )

    print(
        "=========================================="
    )

    print(
        "Local:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "LAN:"
    )

    print(
        "http://192.168.120.147:5000"
    )

    print()

    print(
        "Waiting for authorized devices..."
    )

    print(
        "=========================================="
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )