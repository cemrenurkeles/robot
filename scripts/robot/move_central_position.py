"""Ramène le bras follower en position centrale"""

import os
import time
from lerobot.motors.feetech import FeetechMotorsBus

from motors_config import MOTORS

PORT = os.environ.get("ROBOT_PORT", "/dev/ttyACM0")

bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
bus.connect()

# Lire toutes les positions (valeurs normalisées : -100-100, gripper 0-100)
for name in MOTORS:
    pos = bus.read("Present_Position", name, normalize=False)
    print(f"  {name}: {pos}")

# Position neutre 2048 = centre (ticks bruts, 0-4096)
for name in MOTORS:
    bus.write("Goal_Position", name, 2048, normalize=False)
time.sleep(2)

input("Entrée pour relâcher le bras...")
bus.disconnect()
print("OK")
