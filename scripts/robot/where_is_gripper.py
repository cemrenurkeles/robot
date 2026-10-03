#!/usr/bin/env python3
"""Lit les angles actuels du follower (lecture seule, aucun mouvement, couple inchangé) et affiche la position
de la pince en centimètres AU-DESSUS DE LA TABLE (modèle corrigé, voir arm_kinematics.py).

Repère : origine sur la table sous l'axe de la base, x vers l'avant, y vers la gauche, z vers le haut.

Usage : FOLLOWER_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/where_is_gripper.py
"""

import json
import os
from pathlib import Path

import numpy as np

from arm_kinematics import ARM_JOINTS, ArmKinematics
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def read_follower_deg() -> dict:
    robot_id = os.environ.get("FOLLOWER_ID", "follower_arm")
    calib_file = Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{robot_id}.json"
    calib = {k: MotorCalibration(**v) for k, v in json.loads(calib_file.read_text()).items()}
    bus = FeetechMotorsBus(
        port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
        motors={j: Motor(calib[j].id, "sts3215", MotorNormMode.DEGREES) for j in ARM_JOINTS},
        calibration={j: calib[j] for j in ARM_JOINTS},
    )
    bus.connect(handshake=False)
    try:
        return bus.sync_read("Present_Position")
    finally:
        bus.disconnect(disable_torque=False)


def main():
    arm = ArmKinematics()
    joints = read_follower_deg()
    x, y, z = arm.gripper_cm(joints)
    print("angles :", {j: round(v, 1) for j, v in joints.items()})
    print(f"pince : {z:.1f} cm au-dessus de la table, à {np.hypot(x, y):.1f} cm de l'axe de la base"
          f"  (x = {x:.1f} avant, y = {y:.1f} gauche)")


if __name__ == "__main__":
    main()
