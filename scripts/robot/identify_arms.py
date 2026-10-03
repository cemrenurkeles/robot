#!/usr/bin/env python3
"""Identifie quel bras (follower/leader) est branché sur quel port série, par la tension
mesurée (follower ~12V, leader ~5V), et affiche les `export` prêts à copier-coller."""

import glob
import sys

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

VOLTAGE_THRESHOLD = 8.5  # entre les ~4.9V leader et les ~12V follower mesurés sur le kit 9


def candidate_ports():
    pattern = "/dev/ttyACM*" if sys.platform.startswith("linux") else "/dev/cu.usbmodem*"
    return sorted(glob.glob(pattern))


def classify_voltage(v: float) -> str:
    return "follower" if v > VOLTAGE_THRESHOLD else "leader"


def read_voltage(port: str) -> float:
    bus = FeetechMotorsBus(
        port=port,
        motors={"j1": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.DEGREES)},
    )
    bus.connect()
    try:
        return bus.read("Present_Voltage", "j1", normalize=False) / 10.0
    finally:
        bus.disconnect(disable_torque=False)  # diagnostic en lecture seule, ne doit pas changer l'état du bras


def main():
    ports = candidate_ports()
    if not ports:
        print("Aucun port série détecté (ttyACM*/cu.usbmodem*). Bras branchés ?")
        return

    found = {}
    for port in ports:
        try:
            v = read_voltage(port)
        except Exception as e:
            print(f"{port} -> erreur : {e}")
            continue
        role = classify_voltage(v)
        print(f"{port} -> {role} ({v:.1f}V)")
        found.setdefault(role, port)

    if "follower" not in found or "leader" not in found:
        print("\nIdentification incomplète (un seul bras détecté ou tensions ambiguës), vérifie les branchements.")
        return

    # Pas de commentaire inline sur les lignes export : un copier-coller multi-lignes
    # bloc des trois lignes d'un coup fait planter zsh dessus ("bad pattern: #").
    print("\nÀ copier-coller (ROBOT_PORT = follower, pour les scripts à un seul bras) :")
    print(f"export FOLLOWER_PORT={found['follower']}")
    print(f"export LEADER_PORT={found['leader']}")
    print(f"export ROBOT_PORT={found['follower']}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        assert classify_voltage(12.0) == "follower"
        assert classify_voltage(4.9) == "leader"
        assert classify_voltage(8.5) == "leader"  # seuil non inclus côté follower
        print("self-test OK")
    else:
        main()
