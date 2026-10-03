"""Scanne les IDs moteurs présents sur chaque port série. Ports par défaut Linux (/dev/ttyACM0/1), surchargeables via FOLLOWER_PORT/LEADER_PORT (macOS : /dev/cu.usbmodem*)"""

import os

from lerobot.motors.feetech import FeetechMotorsBus

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/ttyACM1")

for port, name in [(FOLLOWER_PORT, "follower"), (LEADER_PORT, "leader")]:
    print(f"\n=== {port} ({name}) ===")
    try:
        result = FeetechMotorsBus.scan_port(port)
        print("scan_port:", result)
    except Exception as e:
        print("scan_port error:", e)
    bus = FeetechMotorsBus(port=port, motors={})
    bus.connect(handshake=False)
    try:
        ids = bus.broadcast_ping()
        print("broadcast_ping:", ids)
    except Exception as e:
        print("broadcast_ping error:", e)
    bus.disconnect()
