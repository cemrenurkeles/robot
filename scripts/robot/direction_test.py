#!/usr/bin/env python3
"""Étape 7 : mesure dans quel sens (et de combien) le vert se déplace dans l'image de la webcam du poignet
quand chaque articulation bouge de +5°, et pour des combinaisons coude + poignet (avancer le bras en gardant
la caméra vers le bas).

Le follower va à la pose d'observation, mesure le centre du vert, puis pour chaque articulation :
+5°, mesure, retour. Aucune descente au-delà de ces petits mouvements. Ctrl+C pour arrêter.

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX [DEVICE=/dev/video0] python3 scripts/robot/direction_test.py [--combos]
  --combos : ne teste que les combinaisons coude + poignet (étape 7b)
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "camera"))
from camera_utils import V4L2Camera  # noqa: E402
from green_detect import HSV_DEFAULTS, MIN_AREA_DEFAULT, find_green_zone, zone_center  # noqa: E402

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: E402

POSE_FILE = Path("datasets/poses/observation.json")
STEP_DEG = 5.0
SINGLE_TESTS = {j: {j: STEP_DEG} for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]}
# avancer le coude en compensant avec le poignet : le bon signe garde la caméra vers le bas (petit déplacement d'image)
COMBO_TESTS = {
    "coude+5 poignet-5": {"elbow_flex": STEP_DEG, "wrist_flex": -STEP_DEG},
    "coude+5 poignet+5": {"elbow_flex": STEP_DEG, "wrist_flex": STEP_DEG},
}
FPS = 30
LOWER = np.array([HSV_DEFAULTS["H min"], HSV_DEFAULTS["S min"], HSV_DEFAULTS["V min"]])
UPPER = np.array([HSV_DEFAULTS["H max"], HSV_DEFAULTS["S max"], HSV_DEFAULTS["V max"]])


def move_to(robot, target, duration=1.5):
    """Interpolation linéaire depuis la pose actuelle, puis attente que le bras se stabilise."""
    start = robot.get_observation()
    steps = max(1, int(duration * FPS))
    for i in range(1, steps + 1):
        a = i / steps
        robot.send_action({k: start[k] + (v - start[k]) * a for k, v in target.items()})
        time.sleep(1 / FPS)
    time.sleep(1.0)


def measure_center(cap, n=10):
    """Centre moyen du vert sur n images (après avoir vidé les images en attente), ou None si pas de vert."""
    for _ in range(5):
        cap.read()
    centers = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOWER, UPPER)
        _, zones, _ = find_green_zone(mask, MIN_AREA_DEFAULT)
        if zones:
            centers.append(zone_center(zones, frame.shape))
    if len(centers) < n // 2:
        return None
    return tuple(np.mean(centers, axis=0))


def main():
    if not POSE_FILE.exists():
        raise SystemExit(f"{POSE_FILE} introuvable : lancer d'abord scripts/robot/save_pose.py")
    pose = json.loads(POSE_FILE.read_text())
    tests = COMBO_TESTS if "--combos" in sys.argv else {**SINGLE_TESTS, **COMBO_TESTS}

    cap = V4L2Camera(device=os.environ.get("DEVICE", "/dev/video0"), width=640, height=480, fps=30)
    cap.open()
    if not cap.isOpened():
        raise SystemExit("Caméra du poignet inaccessible (DEVICE=/dev/video1 ?)")

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
            id=os.environ.get("FOLLOWER_ID", "follower_arm"),
            disable_torque_on_disconnect=False,
            max_relative_target=10.0,
        )
    )
    robot.connect(calibrate=False)
    results = {}
    try:
        print("Retour à la pose d'observation...")
        move_to(robot, pose, duration=2.5)
        ref = measure_center(cap)
        if ref is None:
            raise SystemExit("Pas de vert visible depuis la pose d'observation : placer l'objet vert dans le champ.")
        print(f"Centre de référence : ({ref[0]:.0f}, {ref[1]:.0f})")

        for name, deltas in tests.items():
            target = dict(pose)
            for joint, delta in deltas.items():
                target[f"{joint}.pos"] += delta
            move_to(robot, target)
            c = measure_center(cap)
            move_to(robot, pose)
            if c is None:
                print(f"  {name:18} : vert perdu de vue")
                continue
            dx, dy = c[0] - ref[0], c[1] - ref[1]
            results[name] = {"deltas_deg": deltas, "dx_px_per_deg": round(dx / STEP_DEG, 2),
                             "dy_px_per_deg": round(dy / STEP_DEG, 2)}
            print(f"  {name:18} : le vert bouge de dx={dx:+5.0f} px, dy={dy:+5.0f} px")
    except KeyboardInterrupt:
        print("\narrêté")
    finally:
        robot.disconnect()
        cap.release()

    if results:
        out = Path("datasets/poses/directions.json")
        previous = json.loads(out.read_text()) if out.exists() else {}
        out.write_text(json.dumps({**previous, **results}, indent=2))
        print(f"\nRésultats (pixels par degré) sauvés dans {out}")
        print("(dx > 0 : le vert part vers la droite de l'image ; dy > 0 : vers le bas)")


if __name__ == "__main__":
    main()
