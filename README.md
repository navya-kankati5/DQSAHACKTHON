# 🔐 DQSA Traffic Classifier

## Dynamic Quantized Self-Attention (DQSA) for Encrypted Network Traffic Classification

A privacy-aware AI framework for classifying encrypted network traffic using network-flow characteristics without inspecting encrypted payload content.

---

## 🚀 Project Overview

Modern network traffic is increasingly encrypted, making traditional payload-based inspection difficult.

This project proposes **Dynamic Quantized Self-Attention (DQSA)** to analyze encrypted network traffic using flow-level characteristics such as:

- Flow duration
- Packet size
- Byte rate
- Packet rate
- Inter-arrival time
- Flow timing statistics

The system performs multi-task classification of network traffic into:

1. **Application Category**
2. **Service Type**
3. **VPN / Non-VPN**

---

## 🧠 Proposed DQSA Architecture

```text
Network Traffic
       ↓
Flow Feature Extraction
       ↓
23 Numerical Features
       ↓
64-D Feature Embedding
       ↓
TGAR
(Task-Gated Attention Router)
       ↓
SQ-SAH
(Soft Quantized Self-Attention Heads)
       ↓
Multi-Task Prediction
    ↙      ↓       ↘
Application Service   VPN
Category    Type    Status
