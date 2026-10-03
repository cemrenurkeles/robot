#!/usr/bin/env python3
"""Déplace une articulation du follower de quelques degrés, en douceur, puis garde la position.

Par défaut : tourne la base (shoulder_pan) de +15°, soit la « tête » du bras un peu à gauche
(vu de face ; delta négatif = à droite).
Les autres articulations restent où elles sont. Le bras garde la pose à la fin (couple maintenu),
sauf avec --release (couple coupé : le bras devient souple et peut retomber, le tenir).

Usage :
  ROBOT_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/move_joint.py              # base +15°
  ROBOT_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/move_joint.py --delta -15  # autre sens
  ROBOT_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/move_joint.py --joint wrist_roll --delta 30
"""

import argparse
import os
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_DELTA = 45.0  # degrés, garde-fou contre une faute de frappe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", default="shoulder_pan", choices=JOINTS)
    parser.add_argument("--delta", type=float, default=15.0, help="degrés (négatif = autre sens)")
    parser.add_argument("--duration", type=float, default=1.5, help="durée du mouvement en secondes")
    parser.add_argument("--release", action="store_true", help="couper le couple à la fin")
    args = parser.parse_args()
    if abs(args.delta) > MAX_DELTA:
        raise SystemExit(f"--delta limité à ±{MAX_DELTA}° par sécurité")

    config = SOFollowerRobotConfig(
        port=os.environ.get("ROBOT_PORT", "/dev/ttyACM0"),
        id=os.environ.get("ROBOT_ID", "follower_arm"),
        disable_torque_on_disconnect=args.release,
        max_relative_target=10.0,  # jamais plus de 10° d'écart par commande
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False)  # pas de calibration interactive : échoue si le fichier ne correspond pas
    try:
        start = robot.get_observation()
        pose = {f"{j}.pos": start[f"{j}.pos"] for j in JOINTS}
        key = f"{args.joint}.pos"
        origin, target = pose[key], pose[key] + args.delta
        print(f"{args.joint} : {origin:.1f}° -> {target:.1f}°")

        fps = 50
        steps = max(1, int(args.duration * fps))
        for i in range(1, steps + 1):
            pose[key] = origin + args.delta * i / steps
            robot.send_action(pose)
            time.sleep(1 / fps)
        # gain P réglé bas par lerobot (anti-tremblement) : le moteur finit sa course lentement
        reached, deadline = robot.get_observation()[key], time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.2)
            now = robot.get_observation()[key]
            if abs(now - reached) < 0.2:
                break
            reached = now
        print(f"position atteinte : {reached:.1f}° (écart {reached - target:+.1f}°)")
    finally:
        robot.disconnect()
    print("couple coupé, bras souple" if args.release else "bras maintenu en position (couple actif)")


if __name__ == "__main__":
    main()
