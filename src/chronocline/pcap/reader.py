"""Optional, offline PCAP timestamp extraction; no packet capture or transmission."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def read_timestamps(path: str | Path) -> dict[str, object]:
    """Read packet timestamps if scapy is installed, otherwise fail gracefully."""
    try:
        from scapy.utils import PcapReader
    except ImportError as error:
        raise RuntimeError("PCAP support requires chronocline[pcap]") from error
    file = Path(path)
    with PcapReader(str(file)) as reader:
        timestamps = np.array([float(packet.time) for packet in reader])
    timestamps.sort()
    return {
        "timestamps": timestamps,
        "inter_arrivals": np.diff(timestamps),
        "duplicate_timestamp_count": int(np.sum(np.diff(timestamps) == 0)),
        "source_hash": hashlib.sha256(file.read_bytes()).hexdigest(),
    }
