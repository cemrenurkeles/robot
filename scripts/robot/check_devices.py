"""Vérifie que chaque port série répond bien (connexion moteur test), PAS que le rôle follower/leader
affiché est le bon : les deux noms viennent juste de FOLLOWER_PORT/LEADER_PORT (ou des valeurs par
défaut /dev/ttyACM0/1), rien ne garantit que ça correspond au bras réel. Pour confirmer le rôle réel,
voir identify_arms.py (classification par tension). Ports par défaut Linux (/dev/ttyACM0/1),
surchargeables via FOLLOWER_PORT/LEADER_PORT (macOS : /dev/cu.usbmodem*)"""

import os

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/ttyACM1")

for port, name in [(FOLLOWER_PORT, "follower"), (LEADER_PORT, "leader")]:
    try:
        bus = FeetechMotorsBus(
            port=port,
            motors={"j1": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.DEGREES)},
        )
        bus.connect()
        print(f"{port} -> {name} OK")
        bus.disconnect(disable_torque=False)  # diagnostic en lecture seule, ne doit pas changer l'état du bras
    except Exception as e:
        print(f"{port} -> {name} ERREUR: {e}")
