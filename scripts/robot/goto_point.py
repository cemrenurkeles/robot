#!/usr/bin/env python3
"""Amène la pince du follower à une hauteur et une distance données (repère table, modèle corrigé), en gardant
son orientation et la direction actuelle de la base. Mouvement lent, butées de la calibration vérifiées avant.
Sert à valider le modèle : mesurer ensuite à la règle.

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/goto_point.py --hauteur 8 --distance 25
    --hauteur  : hauteur du bout de la pince au-dessus de la table (cm)
    --distance : distance horizontale entre l'axe de la base et le bout de la pince (cm)
"""

import argparse
import os
import time

import numpy as np

from arm_kinematics import ARM_JOINTS, ArmKinematics
from find_green import joint_limits
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

SPEED_DEG_S = 15.0
FPS = 30


def move_joints(robot, target, speed=SPEED_DEG_S):
    """Interpolation linéaire en angles, vitesse limitée, puis stabilisation."""
    start = robot.get_observation()
    span = max(abs(v - start[k]) for k, v in target.items())
    steps = max(FPS, int(span / speed * FPS))
    for i in range(1, steps + 1):
        a = i / steps
        robot.send_action({k: start[k] + (v - start[k]) * a for k, v in target.items()})
        time.sleep(1 / FPS)
    time.sleep(1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hauteur", type=float, required=True)
    parser.add_argument("--distance", type=float, required=True)
    args = parser.parse_args()
    if not 2.0 <= args.hauteur <= 30.0 or not 10.0 <= args.distance <= 45.0:
        raise SystemExit("hors des plages autorisées : hauteur 2-30 cm, distance 10-45 cm")

    arm = ArmKinematics()
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
            id=os.environ.get("FOLLOWER_ID", "follower_arm"),
            disable_torque_on_disconnect=False,
            max_relative_target=10.0,
        )
    )
    robot.connect(calibrate=False)
    try:
        obs = robot.get_observation()
        x, y, z = arm.gripper_cm(obs)
        heading = np.array([x, y]) / np.hypot(x, y)
        target = np.array([*(heading * args.distance), args.hauteur])
        print(f"pince actuelle : hauteur {z:.1f} cm, distance {np.hypot(x, y):.1f} cm")
        print(f"cible          : hauteur {args.hauteur:.1f} cm, distance {args.distance:.1f} cm")

        solution, err = arm.solve(obs, target)
        if err > 0.5:
            raise SystemExit(f"position impossible à atteindre en gardant l'orientation (erreur {err:.1f} cm)")
        limits = joint_limits(robot.id)
        for j in ARM_JOINTS:
            lo, hi = limits[f"{j}.pos"]
            if not lo <= solution[f"{j}.pos"] <= hi:
                raise SystemExit(f"{j} sortirait de ses butées ({solution[f'{j}.pos']:.1f}° hors {lo:.0f}..{hi:.0f}°)")
        print("angles :", {j: f"{obs[f'{j}.pos']:.1f} -> {solution[f'{j}.pos']:.1f}" for j in ARM_JOINTS})

        target_action = dict(obs)
        target_action.update(solution)
        move_joints(robot, target_action)

        reached = robot.get_observation()
        x, y, z = arm.gripper_cm(reached)
        print(f"atteint (selon le modèle) : hauteur {z:.1f} cm, distance {np.hypot(x, y):.1f} cm")
        print("-> mesure à la règle la hauteur et la distance réelles")
    except KeyboardInterrupt:
        print("\narrêté")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
