#!/usr/bin/env python3
"""Étapes 8 et 9 : balayage pour trouver le vert avec la webcam du poignet, puis centrage au-dessus.

Balayage : depuis la pose d'observation, la base (shoulder_pan) tourne par pas de 8° : d'abord vers la gauche
jusqu'à -40°, puis vers la droite jusqu'à +40°. Arrêt dès que du vert est vu.
Centrage : petites corrections en boucle jusqu'à ce que le centre du vert soit au centre de l'image :
  gauche/droite -> base ; haut/bas -> coude + et poignet - ensemble (avance le bras, caméra toujours vers le bas).
Aucune descente volontaire. Fenêtre caméra en direct (Q ou Ctrl+C pour arrêter).
Sécurités du centrage : jamais au-delà des butées de la calibration (arrêt si une articulation y arrive),
arrêt si l'écart au centre grandit deux tours de suite.

Après le centrage, le bras avance encore de AVANCE_DEG degrés (combinaison coude/poignet) : 2° par défaut,
réglable avec --avance (négatif = recule).

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX [DEVICE=/dev/video0] python3 scripts/robot/find_green.py [--avance 2]
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
SCAN_STEP_DEG = 8.0
SCAN_RANGE_DEG = 40.0
PAN_SPEED_DEG_S = 20.0  # vitesse de rotation de la base pendant le balayage
FPS = 30
LOWER = np.array([HSV_DEFAULTS["H min"], HSV_DEFAULTS["S min"], HSV_DEFAULTS["V min"]])
UPPER = np.array([HSV_DEFAULTS["H max"], HSV_DEFAULTS["S max"], HSV_DEFAULTS["V max"]])
WIN = "find_green  |  Q=arreter"

# Centrage (valeurs mesurées par direction_test.py ; la boucle corrige les imprécisions)
PAN_PX_PER_DEG = 6.0  # base +1° -> le vert part d'environ 6 px vers la gauche
REACH_PX_PER_DEG = 7.0  # coude +1° et poignet -1° -> le vert part d'environ 7 px vers le bas
GAIN = 0.5  # ne corrige que la moitié de l'écart estimé à chaque tour (approche sans dépasser)
MAX_STEP_DEG = 3.0  # correction maximale par tour et par commande
TOLERANCE_PX = 20  # centré quand le vert est à moins de 20 px du centre de l'image
MAX_ITER = 20
MAX_PAN_OFFSET = 50.0  # écart maximal de la base autour de la pose d'observation
MAX_REACH_OFFSET = 20.0  # écart maximal de la combinaison coude/poignet
LIMIT_MARGIN_DEG = 2.0  # on reste à 2° des butées de la calibration
AVANCE_DEG = 2.0  # avance finale après centrage (coude +, poignet -) ; signe à inverser si le bras recule


def joint_limits(robot_id):
    """Butées (min, max) en degrés de chaque articulation, d'après le fichier de calibration lerobot.
    Même conversion que lerobot en mode degrés : (valeur - milieu) * 360 / 4095."""
    path = Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{robot_id}.json"
    limits = {}
    for joint, c in json.loads(path.read_text()).items():
        mid = (c["range_min"] + c["range_max"]) / 2
        to_deg = lambda raw: (raw - mid) * 360 / 4095  # noqa: E731
        limits[f"{joint}.pos"] = (to_deg(c["range_min"]) + LIMIT_MARGIN_DEG, to_deg(c["range_max"]) - LIMIT_MARGIN_DEG)
    return limits


class Stop(Exception):
    pass


def detect(frame):
    """Renvoie (cx, cy) du vert ou None, et dessine la détection sur l'image."""
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOWER, UPPER)
    _, zones, _ = find_green_zone(mask, MIN_AREA_DEFAULT)
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    if not zones:
        return None
    x, y, bw, bh = cv2.boundingRect(np.vstack(zones))
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
    cx, cy = zone_center(zones, frame.shape)
    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
    return cx, cy


def show(cap, status):
    """Lit une image, détecte, affiche ; renvoie le centre ou None. Lève Stop si Q est pressé."""
    ok, frame = cap.read()
    if not ok:
        raise Stop("pas d'image de la caméra")
    center = detect(frame)
    cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        raise Stop("arrêté (Q)")
    return center


def move_to(robot, cap, target, status):
    """Interpolation linéaire vers la cible (vitesse limitée), caméra affichée pendant le mouvement."""
    start = robot.get_observation()
    span = max(abs(v - start[k]) for k, v in target.items())
    steps = max(FPS, int(span / PAN_SPEED_DEG_S * FPS))
    for i in range(1, steps + 1):
        a = i / steps
        robot.send_action({k: start[k] + (v - start[k]) * a for k, v in target.items()})
        show(cap, status)
    t_end = time.time() + 0.8  # stabilisation du bras avant de regarder
    while time.time() < t_end:
        show(cap, status)


def look(cap, status, n=8):
    """Regarde n images ; renvoie le centre moyen si le vert est vu sur au moins la moitié, sinon None."""
    centers = [c for c in (show(cap, status) for _ in range(n)) if c is not None]
    return tuple(int(v) for v in np.mean(centers, axis=0)) if len(centers) >= n // 2 else None


def reach_bounds(pose, limits):
    """Plage autorisée pour la combinaison (coude + r, poignet - r) : butées des deux articulations et MAX_REACH_OFFSET."""
    e_lo, e_hi = limits["elbow_flex.pos"]
    w_lo, w_hi = limits["wrist_flex.pos"]
    e0, w0 = pose["elbow_flex.pos"], pose["wrist_flex.pos"]
    lo = max(-MAX_REACH_OFFSET, e_lo - e0, w0 - w_hi)
    hi = min(MAX_REACH_OFFSET, e_hi - e0, w0 - w_lo)
    return lo, hi


def center_above(robot, cap, cmd, pose, limits):
    """Boucle de centrage. cmd = pose commandée actuelle (modifiée sur place). Renvoie True si centré."""
    pan0 = pose["shoulder_pan.pos"]
    pan_lo = max(pan0 - MAX_PAN_OFFSET, limits["shoulder_pan.pos"][0])
    pan_hi = min(pan0 + MAX_PAN_OFFSET, limits["shoulder_pan.pos"][1])
    reach_lo, reach_hi = reach_bounds(pose, limits)
    print(f"  marges : base {pan_lo - pan0:+.0f}°/{pan_hi - pan0:+.0f}°, avancer/reculer {reach_lo:+.0f}°/{reach_hi:+.0f}°")
    reach = 0.0
    errors = []
    for i in range(1, MAX_ITER + 1):
        status = f"centrage {i}/{MAX_ITER}"
        c = look(cap, status)
        if c is None:
            print("vert perdu de vue : arrêt du centrage")
            return False
        frame_w, frame_h = 640, 480
        ex, ey = c[0] - frame_w // 2, c[1] - frame_h // 2
        print(f"  tour {i:2} : centre = {c}, écart = ({ex:+4d}, {ey:+4d}) px")
        if abs(ex) < TOLERANCE_PX and abs(ey) < TOLERANCE_PX:
            return True
        errors.append(float(np.hypot(ex, ey)))
        if len(errors) >= 3 and errors[-1] > errors[-2] > errors[-3]:
            print("l'écart grandit deux tours de suite (correction dans le mauvais sens ?) : arrêt du centrage")
            return False
        d_pan = float(np.clip(GAIN * ex / PAN_PX_PER_DEG, -MAX_STEP_DEG, MAX_STEP_DEG))
        d_reach = float(np.clip(-GAIN * ey / REACH_PX_PER_DEG, -MAX_STEP_DEG, MAX_STEP_DEG))
        new_pan = float(np.clip(cmd["shoulder_pan.pos"] + d_pan, pan_lo, pan_hi))
        new_reach = float(np.clip(reach + d_reach, reach_lo, reach_hi))
        pan_blocked = abs(ex) >= TOLERANCE_PX and new_pan == cmd["shoulder_pan.pos"]
        reach_blocked = abs(ey) >= TOLERANCE_PX and new_reach == reach
        if pan_blocked or reach_blocked:
            which = "base" if pan_blocked else "coude/poignet"
            print(f"butée atteinte ({which}) : impossible d'aller plus loin, arrêt du centrage")
            return False
        cmd["shoulder_pan.pos"] = new_pan
        cmd["elbow_flex.pos"] = pose["elbow_flex.pos"] + new_reach
        cmd["wrist_flex.pos"] = pose["wrist_flex.pos"] - new_reach
        reach = new_reach
        move_to(robot, cap, dict(cmd), status)
    print("nombre maximal de corrections atteint")
    return False


def main():
    if not POSE_FILE.exists():
        raise SystemExit(f"{POSE_FILE} introuvable : lancer d'abord scripts/robot/save_pose.py")
    pose = json.loads(POSE_FILE.read_text())
    pan0 = pose["shoulder_pan.pos"]

    cap = V4L2Camera(device=os.environ.get("DEVICE", "/dev/video0"), width=640, height=480, fps=30)
    cap.open()
    if not cap.isOpened():
        raise SystemExit("Caméra du poignet inaccessible (DEVICE=/dev/video1 ?)")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 720)

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
            id=os.environ.get("FOLLOWER_ID", "follower_arm"),
            disable_torque_on_disconnect=False,
            max_relative_target=10.0,
        )
    )
    robot.connect(calibrate=False)

    n = int(SCAN_RANGE_DEG // SCAN_STEP_DEG)
    offsets = [0.0] + [-SCAN_STEP_DEG * i for i in range(1, n + 1)] + [SCAN_STEP_DEG * i for i in range(1, n + 1)]
    found = None
    try:
        for off in offsets:
            target = dict(pose)
            target["shoulder_pan.pos"] = pan0 + off
            status = f"balayage : base {off:+.0f} deg"
            print(status)
            move_to(robot, cap, target, status)
            center = look(cap, status)
            if center is not None:
                found = (off, center)
                break
        if found:
            off, (cx, cy) = found
            print(f"VERT TROUVE : base {off:+.0f}° autour de la pose ({pan0 + off:.1f}°), centre = ({cx}, {cy})")
            cmd = dict(pose)
            cmd["shoulder_pan.pos"] = pan0 + off
            if center_above(robot, cap, cmd, pose, joint_limits(robot.id)):
                print("CENTRE AU-DESSUS DU VERT")
                avance = float(sys.argv[sys.argv.index("--avance") + 1]) if "--avance" in sys.argv else AVANCE_DEG
                if avance:
                    lo, hi = reach_bounds(pose, joint_limits(robot.id))
                    reach = cmd["elbow_flex.pos"] - pose["elbow_flex.pos"]
                    new_reach = float(np.clip(reach + avance, lo, hi))
                    cmd["elbow_flex.pos"] = pose["elbow_flex.pos"] + new_reach
                    cmd["wrist_flex.pos"] = pose["wrist_flex.pos"] - new_reach
                    move_to(robot, cap, dict(cmd), f"avance de {new_reach - reach:+.1f} deg")
                    note = "" if abs(new_reach - reach - avance) < 1e-6 else " (limité par une butée)"
                    print(f"AVANCE de {new_reach - reach:+.1f}° (coude {new_reach - reach:+.1f}°, poignet {reach - new_reach:+.1f}°){note}")
                print("pose :", {k.removesuffix(".pos"): round(v, 1) for k, v in robot.get_observation().items()})
                t_end = time.time() + 3  # montre le résultat
                while time.time() < t_end:
                    show(cap, "CENTRE AU-DESSUS DU VERT")
        else:
            print("PAS DE VERT DANS LA ZONE : retour à la pose d'observation")
            move_to(robot, cap, pose, "pas de vert : retour")
    except (Stop, KeyboardInterrupt) as e:
        print(f"\n{e or 'arrêté (Ctrl+C)'}")
    finally:
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
