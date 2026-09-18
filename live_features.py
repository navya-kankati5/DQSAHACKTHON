import time
import socket
from collections import defaultdict

from scapy.all import sniff, IP, TCP, UDP


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

    # Approximate 23 features
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


def print_flows():

    print("\nLIVE FLOWS")
    print("-" * 90)

    for key in list(flows.keys()):

        features = get_features(key)

        src, dst, src_port, dst_port, protocol = key

        print(
            f"{src}:{src_port} -> "
            f"{dst}:{dst_port} "
            f"{protocol} | "
            f"packets={flows[key]['packets']} | "
            f"bytes={flows[key]['bytes']}"
        )

        print(
            "23 features:",
            len(features)
        )


print("Starting live packet capture for 15 seconds...")
print("Open Chrome and visit Google/YouTube now.\n")

sniff(
    prn=process_packet,
    store=False,
    timeout=15
)

print("\nCapture finished.")
print_flows()
print("Generate traffic by opening Chrome or visiting a website.")
print("Press CTRL+C to stop.\n")

try:
    sniff(prn=process_packet, store=False)
except KeyboardInterrupt:
    pass

print("\nCapture stopped.")
print_flows()