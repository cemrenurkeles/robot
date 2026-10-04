#!/usr/bin/env python3
"""Étapes 8 et 9 : balayage pour trouver le vert avec la webcam du poignet, puis centrage au-dessus.

Balayage : depuis la pose d'observation, la base (shoulder_pan) tourne par pas de 8° : d'abord vers la gauche
jusqu'à -40°, puis vers la droite jusqu'à +40°. Arrêt dès que du vert est vu.
Centrage : la pince se déplace en centimètres sur la table (cinématique, hauteur et orientation constantes) jusqu'à
ce que le centre du vert soit sur le pixel « sous la pince » (mesuré une fois avec --calibrer-pince, enregistré
dans datasets/poses/cible_pince.json ; croix jaune). La correspondance pixels <-> cm est mesurée au début du
centrage par deux petits déplacements de 1,5 cm (avant, gauche).
Aucune descente volontaire. Fenêtre caméra en direct (Q ou Ctrl+C pour arrêter).
Sécurités du centrage : butées de la calibration vérifiées avant chaque mouvement, arrêt si l'écart grandit deux
tours de suite, si le vert est perdu, ou s'il faudrait s'éloigner de plus de 30 cm.

Étape 10c, après le centrage (cinématique corrigée, voir arm_kinematics.py) :
  1. ajustement optionnel : --avance cm vers l'avant, --lateral cm sur le côté (0 par défaut, le centrage vise
     déjà le point sous la pince) ;
  2. la pince s'ouvre (--ouverture, 30 par défaut sur l'échelle 0-100 de lerobot, 0 = fermée) ;
  3. elle descend tout droit par pas de 2 cm jusqu'à --hauteur cm au-dessus de la table (3,5 par défaut) ;
  4. la pince se ferme (sur l'objet) ;
  5. la pince remonte tout droit à sa hauteur de départ, puis le bras revient à la pose d'observation en
     gardant la pince fermée (sauf --sans-retour).
  Chaque position est corrigée en deuxième passe (les moteurs s'arrêtent un peu avant la cible).
  Contact : si un moteur force ou décroche de sa consigne avant l'arrivée, arrêt et remontée de 5 mm.

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX [DEVICE=/dev/video0] python3 scripts/robot/find_green.py \
      [--avance 0] [--lateral 0] [--hauteur 3.5] [--ouverture 30] [--sans-descente] [--sans-retour]
  --lateral : > 0 = vers la gauche du robot, < 0 = vers sa droite (vu de derrière le robot)
  --calibrer-pince : va à la pose d'observation ; poser l'objet vert juste sous le bout de la pince, puis
                     Entrée (ou C) dans la fenêtre : le pixel « sous la pince » est mesuré et enregistré.
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

from arm_kinematics import ArmKinematics  # noqa: E402
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: E402

POSE_FILE = Path("datasets/poses/observation.json")
SCAN_STEP_DEG = 8.0
SCAN_RANGE_DEG = 40.0
PAN_SPEED_DEG_S = 20.0  # vitesse de rotation de la base pendant le balayage
FPS = 30
LOWER = np.array([HSV_DEFAULTS["H min"], HSV_DEFAULTS["S min"], HSV_DEFAULTS["V min"]])
UPPER = np.array([HSV_DEFAULTS["H max"], HSV_DEFAULTS["S max"], HSV_DEFAULTS["V max"]])
WIN = "find_green  |  Q=arreter"

# Centrage en centimètres (cinématique) : la correspondance pixels <-> cm est mesurée au début de chaque centrage
PROBE_CM = 1.5  # petits déplacements de mesure (avant, gauche)
GAIN = 0.7  # fraction de l'écart estimé corrigée à chaque tour (approche sans dépasser)
MAX_STEP_CM = 3.0  # déplacement maximal par tour
MAX_OFFSET_CM = 30.0  # déplacement total maximal autour du point de départ du centrage (était 15)
TOLERANCE_CM = 0.5  # centré quand l'écart, converti en cm sur la table, est sous 0,5 cm
CENTERING_IK_TOL_CM = 1.5  # pendant le centrage, la boucle rattrape les petites erreurs de position
MAX_ITER = 20
LIMIT_MARGIN_DEG = 2.0  # on reste à 2° des butées de la calibration
# Cible du centrage : pixel de l'image qui se trouve sous le bout de la pince (--calibrer-pince)
TARGET_FILE = Path("datasets/poses/cible_pince.json")
TARGET_PX = (320, 240)  # remplacé au démarrage par cible_pince.json s'il existe

# Avance et descente (étape 10c)
AVANCE_CM = 0.0  # ajustements optionnels après centrage (le centrage vise déjà le point sous la pince)
LATERAL_CM = 0.0  # > 0 : vers la gauche du robot (vu de derrière) ; < 0 : vers sa droite
HAUTEUR_FINALE_CM = 3.5  # hauteur du bout de la pince au-dessus de la table à la fin de la descente
HAUTEUR_MIN_CM = 2.0  # jamais plus bas (marge d'erreur du modèle ~1-2 cm)
DESCENT_STEP_CM = 2.0  # pas de descente (la détection de contact réagit à la fin de chaque pas)
RETREAT_CM = 0.5  # remontée après un contact
CONTACT_ERR_DEG = 5.0  # contact si l'écart consigne/position augmente de plus de 5° sur épaule/coude/poignet
CONTACT_LOAD = 300  # ... ou si la charge d'un de ces moteurs augmente de plus de 300 (‰ du couple max)
PITCH_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
GRIPPER_OPEN = 30.0  # ouverture avant la descente (0 = fermée ; les démonstrations allaient jusqu'à ~20-23)
GRIPPER_CLOSED = 0.0  # fermeture après la descente : le couple de la pince est limité par lerobot (configure)


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
    cv2.drawMarker(frame, TARGET_PX, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 2)  # cible « sous la pince »
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


def center_above(robot, cap, cmd, limits):
    """Centrage en cm : la pince se déplace en ligne droite sur la table (hauteur et orientation constantes)
    jusqu'à ce que le vert soit sur la cible TARGET_PX. cmd = consigne actuelle (mise à jour). Renvoie True si centré.
    La correspondance pixels <-> cm est mesurée sur place par deux petits déplacements (avant, gauche)."""
    arm = ArmKinematics()
    p0 = arm.gripper_cm(robot.get_observation())
    heading = p0[:2] / np.linalg.norm(p0[:2])
    left = np.array([-heading[1], heading[0]])

    def point(offset):  # offset = (cm vers l'avant, cm vers la gauche du robot), hauteur de départ
        return np.array([*(p0[:2] + heading * offset[0] + left * offset[1]), p0[2]])

    def target_error(c):
        return np.array([c[0] - TARGET_PX[0], c[1] - TARGET_PX[1]], dtype=float)

    c0 = look(cap, "mesure pixels/cm")
    if c0 is None:
        print("vert perdu de vue : arrêt du centrage")
        return False
    columns = []
    for name, probe in (("avant", (PROBE_CM, 0.0)), ("gauche", (0.0, PROBE_CM))):
        go_to_point(robot, cap, arm, limits, cmd, point(probe), f"mesure : {PROBE_CM} cm vers {name}")
        c = look(cap, "mesure pixels/cm")
        go_to_point(robot, cap, arm, limits, cmd, point((0.0, 0.0)), "retour")
        if c is None:
            print(f"vert perdu pendant la mesure vers {name} : arrêt du centrage")
            return False
        columns.append((np.array(c, float) - np.array(c0, float)) / PROBE_CM)
    J = np.column_stack(columns)  # pixels par cm : colonne 0 = avant, colonne 1 = gauche
    print(f"  1 cm vers l'avant -> ({J[0, 0]:+.0f}, {J[1, 0]:+.0f}) px ; 1 cm vers la gauche -> ({J[0, 1]:+.0f}, {J[1, 1]:+.0f}) px")
    if abs(np.linalg.det(J)) < 20:
        print("mesure pixels/cm inexploitable (le vert ne bouge pas assez) : arrêt du centrage")
        return False

    offset = np.zeros(2)
    errors = []
    for i in range(1, MAX_ITER + 1):
        status = f"centrage {i}/{MAX_ITER}"
        c = look(cap, status)
        if c is None:
            print("vert perdu de vue : arrêt du centrage")
            return False
        e = target_error(c)
        e_cm = np.linalg.solve(J, e)  # écart converti en cm (avant, gauche)
        print(f"  tour {i:2} : centre = {c}, écart = ({e[0]:+4.0f}, {e[1]:+4.0f}) px = {np.linalg.norm(e_cm):.1f} cm")
        if np.linalg.norm(e_cm) < TOLERANCE_CM:
            return True
        errors.append(float(np.linalg.norm(e_cm)))
        if len(errors) >= 3 and errors[-1] > errors[-2] > errors[-3]:
            print("l'écart grandit deux tours de suite : arrêt du centrage")
            return False
        step = -GAIN * e_cm
        step *= min(1.0, MAX_STEP_CM / max(np.linalg.norm(step), 1e-9))
        offset = offset + step
        if np.linalg.norm(offset) > MAX_OFFSET_CM:
            print(f"déplacement de plus de {MAX_OFFSET_CM:.0f} cm nécessaire : arrêt du centrage")
            return False
        try:
            go_to_point(robot, cap, arm, limits, cmd, point(offset), status, ik_tol_cm=CENTERING_IK_TOL_CM)
        except Stop as e:
            if "(Q)" in str(e):
                raise
            print(f"objet hors de portée du bras ({e}) : arrêt du centrage")
            return False
    print("nombre maximal de corrections atteint")
    return False


def go_to_point(robot, cap, arm, limits, cmd, target_cm, status, passes=2, ik_tol_cm=0.5):
    """Amène la pince en target_cm (repère table) en gardant son orientation. cmd = consigne actuelle (mise à jour).
    Deuxième passe : si le bras s'est arrêté avant la cible, on vise d'autant plus loin. Renvoie la position atteinte."""
    target_cm = np.asarray(target_cm, dtype=float)
    aim = target_cm.copy()
    for _ in range(passes):
        solution, err = arm.solve(cmd, aim)
        if err > ik_tol_cm:
            raise Stop(f"position impossible à atteindre en gardant l'orientation (erreur {err:.1f} cm)")
        for k, v in solution.items():
            lo, hi = limits[k]
            if not lo <= v <= hi:
                raise Stop(f"{k.removesuffix('.pos')} sortirait de ses butées ({v:.1f}° hors {lo:.0f}..{hi:.0f}°)")
        cmd.update(solution)
        move_to(robot, cap, dict(cmd), status)
        reached = arm.gripper_cm(robot.get_observation())
        miss = target_cm - reached
        if np.linalg.norm(miss) < 0.3:
            break
        aim = aim + np.clip(miss, -2.0, 2.0)
    return reached


def contact_signals(robot, cmd):
    """(plus grand écart consigne/position en degrés, charges) sur épaule, coude, poignet."""
    obs = robot.get_observation()
    err = max(abs(cmd[f"{j}.pos"] - obs[f"{j}.pos"]) for j in PITCH_JOINTS)
    loads = robot.bus.sync_read("Present_Load", PITCH_JOINTS, normalize=False)
    return err, loads


def set_gripper(robot, cap, cmd, value, status):
    """Ouvre/ferme la pince (seul ce moteur bouge), puis laisse le temps de finir."""
    cmd["gripper.pos"] = value
    move_to(robot, cap, dict(cmd), status)


def advance_and_descend(robot, cap, cmd, avance_cm, lateral_cm, hauteur_cm, descendre, ouverture=GRIPPER_OPEN):
    """Étape 10c : avance horizontale (+ décalage latéral caméra -> pince) puis descente verticale avec contact."""
    arm = ArmKinematics()
    limits = joint_limits(robot.id)
    start = arm.gripper_cm(robot.get_observation())
    heading = start[:2] / np.linalg.norm(start[:2])
    left = np.array([-heading[1], heading[0]])  # perpendiculaire, vers la gauche du robot
    print(f"pince : {start[2]:.1f} cm au-dessus de la table, à {np.hypot(*start[:2]):.1f} cm de la base")

    above = np.array([*(start[:2] + heading * avance_cm + left * lateral_cm), start[2]])
    reached = go_to_point(robot, cap, arm, limits, cmd, above, f"avance de {avance_cm:.1f} cm")
    print(f"AVANCE de {avance_cm:.1f} cm, décalage de {lateral_cm:+.1f} cm sur le côté : pince à {reached[2]:.1f} cm"
          f" de haut, {np.hypot(*reached[:2]):.1f} cm de la base")
    if not descendre:
        return

    set_gripper(robot, cap, cmd, ouverture, "ouverture de la pince")
    print(f"PINCE OUVERTE ({ouverture:.0f})")
    err0, load0 = contact_signals(robot, cmd)
    z = reached[2]
    while z > hauteur_cm + 1e-6:
        z = max(hauteur_cm, z - DESCENT_STEP_CM)
        reached = go_to_point(robot, cap, arm, limits, cmd, [*above[:2], z], f"descente : {z:.1f} cm", passes=1)
        err, load = contact_signals(robot, cmd)
        d_load = max(abs(load[j] - load0[j]) for j in PITCH_JOINTS)
        print(f"  descente : visé {z:4.1f} cm, atteint {reached[2]:4.1f} cm | écart moteurs {err:4.1f}° | charge +{d_load}")
        if err > err0 + CONTACT_ERR_DEG or d_load > CONTACT_LOAD:
            print("CONTACT détecté : arrêt de la descente et remontée de 5 mm")
            go_to_point(robot, cap, arm, limits, cmd, [*above[:2], reached[2] + RETREAT_CM], "remontée", passes=1)
            break
    else:
        reached = go_to_point(robot, cap, arm, limits, cmd, [*above[:2], hauteur_cm], "ajustement final")
        print(f"DESCENTE TERMINEE : pince à {reached[2]:.1f} cm au-dessus de la table (visé {hauteur_cm:.1f} cm)")
    set_gripper(robot, cap, cmd, GRIPPER_CLOSED, "fermeture de la pince")
    print(f"PINCE FERMEE (position {robot.get_observation()['gripper.pos']:.1f})")
    if "--sans-retour" in sys.argv:
        return
    # remontée verticale d'abord (on ne traîne pas l'objet sur la table), pince fermée
    reached = go_to_point(robot, cap, arm, limits, cmd, above, "remontée avec l'objet", passes=1)
    print(f"REMONTEE : pince à {reached[2]:.1f} cm au-dessus de la table")


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def calibrate_gripper_target(robot, cap, pose):
    """Va à la pose d'observation, attend que l'objet vert soit posé sous la pince, mesure et enregistre son pixel."""
    move_to(robot, cap, pose, "pose d'observation")
    print("Pose l'objet vert sur la table JUSTE SOUS le bout de la pince (vérifie de côté),")
    print("puis appuie sur Entrée ou C dans la fenêtre de la caméra.")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise Stop("pas d'image de la caméra")
        detect(frame)
        cv2.putText(frame, "objet sous la pince puis Entree / C", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise Stop("arrêté (Q)")
        if key in (13, 10, ord("c")):
            break
    center = look(cap, "mesure", n=20)
    if center is None:
        raise Stop("pas de vert visible : recommencer avec l'objet bien dans le champ")
    TARGET_FILE.write_text(json.dumps({"pixel": list(center), "pose": str(POSE_FILE)}, indent=2))
    print(f"CIBLE PINCE enregistrée : pixel {center} -> {TARGET_FILE}")


def main():
    global TARGET_PX
    if TARGET_FILE.exists():
        TARGET_PX = tuple(json.loads(TARGET_FILE.read_text())["pixel"])
        print(f"cible du centrage : pixel {TARGET_PX} (sous la pince)")
    elif "--calibrer-pince" not in sys.argv:
        print(f"ATTENTION : {TARGET_FILE} absent, le centrage vise le centre de l'image (lancer --calibrer-pince)")
    avance_cm = arg("--avance", AVANCE_CM)
    lateral_cm = arg("--lateral", LATERAL_CM)
    ouverture = arg("--ouverture", GRIPPER_OPEN)
    hauteur_cm = arg("--hauteur", HAUTEUR_FINALE_CM)
    if not 0.0 <= avance_cm <= 10.0 or abs(lateral_cm) > 10.0 or hauteur_cm < HAUTEUR_MIN_CM or not 0 < ouverture <= 100:
        raise SystemExit(f"--avance 0-10 cm, --lateral -10..10 cm, --hauteur ≥ {HAUTEUR_MIN_CM} cm, --ouverture 1-100")
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

    if "--calibrer-pince" in sys.argv:
        try:
            calibrate_gripper_target(robot, cap, pose)
        except (Stop, KeyboardInterrupt) as e:
            print(f"\n{e or 'arrêté (Ctrl+C)'}")
        finally:
            robot.disconnect()
            cap.release()
            cv2.destroyAllWindows()
        return

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
            if center_above(robot, cap, cmd, joint_limits(robot.id)):
                print("CENTRE AU-DESSUS DU VERT")
                descendre = "--sans-descente" not in sys.argv
                advance_and_descend(robot, cap, cmd, avance_cm, lateral_cm, hauteur_cm, descendre, ouverture)
                if descendre and "--sans-retour" not in sys.argv:
                    home = dict(pose)
                    home["gripper.pos"] = GRIPPER_CLOSED  # on garde l'objet
                    move_to(robot, cap, home, "retour à la position initiale")
                    print("RETOUR à la pose d'observation, pince fermée")
                print("pose :", {k.removesuffix(".pos"): round(v, 1) for k, v in robot.get_observation().items()})
                t_end = time.time() + 3  # montre le résultat
                while time.time() < t_end:
                    show(cap, "termine")
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
