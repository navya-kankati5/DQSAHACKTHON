import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/traffic_data.csv")

print("Dataset shape:", df.shape)

# Features
X = df.select_dtypes(include=["number"])

# Application category
df["application_category"] = df["traffic_type"].str.replace("VPN-", "", regex=False)

# Encryption
df["encryption_type"] = df["traffic_type"].apply(
    lambda x: "VPN" if x.startswith("VPN-") else "Non-VPN"
)

# Service type
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

# Train/test split
X_train, X_test, y_train_app, y_test_app = train_test_split(
    X,
    df["application_category"],
    test_size=0.2,
    random_state=42,
    stratify=df["application_category"]
)

# Train Random Forest
model_app = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Application Classifier...")
model_app.fit(X_train, y_train_app)

pred_app = model_app.predict(X_test)

print("\n===== APPLICATION CLASSIFICATION =====")
print("Accuracy:", accuracy_score(y_test_app, pred_app))
print("F1 Score:", f1_score(y_test_app, pred_app, average="weighted"))

# Encryption model
X_train2, X_test2, y_train_enc, y_test_enc = train_test_split(
    X,
    df["encryption_type"],
    test_size=0.2,
    random_state=42,
    stratify=df["encryption_type"]
)

model_enc = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Encryption Classifier...")
model_enc.fit(X_train2, y_train_enc)

pred_enc = model_enc.predict(X_test2)

print("\n===== ENCRYPTION CLASSIFICATION =====")
print("Accuracy:", accuracy_score(y_test_enc, pred_enc))
print("F1 Score:", f1_score(y_test_enc, pred_enc, average="weighted"))

print("\n===== PROJECT BASELINE COMPLETE =====")