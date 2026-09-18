import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("data/traffic_data.csv")


# =========================================================
# 2. CREATE LABELS
# =========================================================

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

df["service_type"] = df["application_category"].map(service_map)


# =========================================================
# 3. FEATURES + CLASSES
# =========================================================

X = df.select_dtypes(include=["number"]).values

app_classes = sorted(df["application_category"].unique())
service_classes = sorted(df["service_type"].unique())
enc_classes = sorted(df["encryption_type"].unique())


# =========================================================
# 4. SCALE DATA
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)


# =========================================================
# 5. DQSA MODEL
# =========================================================

class TaskGatedAttention(nn.Module):

    def __init__(self, d_model=64, heads=4, num_tasks=3):

        super().__init__()

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
        )

        scores = scores / np.sqrt(self.head_dim)

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

        head_output = head_output * gates

        head_output = head_output.transpose(
            1,
            2
        ).contiguous()

        head_output = head_output.view(
            batch_size,
            seq_len,
            self.heads * self.head_dim
        )

        return self.output(head_output)


class SoftQuantization(nn.Module):

    def __init__(self, levels=7):

        super().__init__()

        self.centers = nn.Parameter(
            torch.linspace(-1.0, 1.0, levels)
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
            0.8 * bounded
            +
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
            d_model=64,
            heads=4,
            num_tasks=3
        )

        self.norm1 = nn.LayerNorm(64)

        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

        self.norm2 = nn.LayerNorm(64)

        self.quantizer = SoftQuantization(
            levels=7
        )

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

        app_output = self.app_head(app_rep)

        service_output = self.service_head(
            service_rep
        )

        enc_output = self.enc_head(enc_rep)

        return (
            app_output,
            service_output,
            enc_output
        )


# =========================================================
# 6. LOAD TRAINED MODEL
# =========================================================

device = torch.device("cpu")

model = DQSA(
    num_features=X.shape[1],
    num_app=len(app_classes),
    num_service=len(service_classes),
    num_enc=len(enc_classes)
)

model.load_state_dict(
    torch.load(
        "models/dqsa_model.pth",
        map_location=device
    )
)

model.eval()


# =========================================================
# 7. SELECT A SAMPLE
# =========================================================

sample_index = 0

sample = X_tensor[
    sample_index
].unsqueeze(0)


# =========================================================
# 8. PREDICTION
# =========================================================

with torch.no_grad():

    app_out, service_out, enc_out = model(
        sample
    )

    app_prediction = torch.argmax(
        app_out,
        dim=1
    ).item()

    service_prediction = torch.argmax(
        service_out,
        dim=1
    ).item()

    enc_prediction = torch.argmax(
        enc_out,
        dim=1
    ).item()


# =========================================================
# 9. DISPLAY RESULT
# =========================================================

print("\n======================================")
print("        DQSA TRAFFIC PREDICTION")
print("======================================")

print(
    "\nApplication :",
    app_classes[app_prediction]
)

print(
    "Service Type:",
    service_classes[service_prediction]
)

print(
    "Encryption  :",
    enc_classes[enc_prediction]
)

print("\n======================================")