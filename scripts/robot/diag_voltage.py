"""Affiche la tension mesurée par chaque moteur du bus follower. Diagnostic alimentation (follower : 12V nominal)"""

import os
from lerobot.motors.feetech import FeetechMotorsBus

from motors_config import MOTORS

FOLLOWER_PORT = os.environ.get("ROBOT_PORT", os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"))

bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=MOTORS)
bus.connect()

print(f"{'Motor':<15} {'Voltage (V)':>12}")
for name in MOTORS:
    v = bus.read("Present_Voltage", name, normalize=False) / 10.0
    print(f"{name:<15} {v:>12.1f}")

bus.disconnect(disable_torque=False)  # diagnostic en lecture seule, ne doit pas changer l'état du bras
