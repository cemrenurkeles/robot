"""Détection de la couleur verte sur la webcam du poignet (Innomaker U20CAM, via V4L2Camera).
Étape 5 : HSV + masque du vert (seuils réglables), nettoyage du masque, ensemble des zones vertes
(morceaux séparés compris) encadré, centre (cx, cy) de l'ensemble affiché sur l'image et dans le terminal.
Clic gauche sur l'objet = pipette (règle les seuils sur sa couleur). Q/Echap pour quitter, P pour afficher les seuils

Usage : [DEVICE=/dev/video0] python3 scripts/camera/green_detect.py
"""

import os
import sys
import time

import cv2
import numpy as np

from camera_utils import V4L2Camera

DEVICE = os.environ.get("DEVICE", "/dev/video0")
WIDTH, HEIGHT, FPS = 640, 480, 30

# Plage HSV du vert dans OpenCV (H : 0-179, S et V : 0-255). Vert pur ≈ H 60.
# Réglée pour inclure le vert clair (peu saturé) ; la table beige (H ≈ 15-25) reste exclue.
HSV_DEFAULTS = {"H min": 28, "H max": 90, "S min": 25, "S max": 255, "V min": 70, "V max": 255}
HSV_MAX = {"H min": 179, "H max": 179, "S min": 255, "S max": 255, "V min": 255, "V max": 255}
MIN_AREA_DEFAULT = 100  # pixels : en dessous, une tache verte est considérée comme parasite
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # petit noyau : garde les objets lointains
PIPETTE_HALF = 5  # pipette : moyenne sur un carré 11x11 autour du clic
PIPETTE_MARGIN = {"H": 12, "S": 60, "V": 70}  # tolérance autour de la couleur mesurée


def apply_pipette(hsv, x, y, win_mask):
    """Mesure la couleur HSV médiane autour de (x, y) et règle les trackbars autour de cette couleur."""
    h, w = hsv.shape[:2]
    patch = hsv[max(0, y - PIPETTE_HALF) : min(h, y + PIPETTE_HALF + 1), max(0, x - PIPETTE_HALF) : min(w, x + PIPETTE_HALF + 1)]
    hm, sm, vm = (int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
    limits = {"H": 179, "S": 255, "V": 255}
    for ch, val in zip("HSV", (hm, sm, vm)):
        cv2.setTrackbarPos(f"{ch} min", win_mask, max(0, val - PIPETTE_MARGIN[ch]))
        cv2.setTrackbarPos(f"{ch} max", win_mask, min(limits[ch], val + PIPETTE_MARGIN[ch]))
    print(f"pipette ({x}, {y}) : couleur HSV = ({hm}, {sm}, {vm})")


def find_green_zone(mask, min_area):
    """Nettoie le masque, garde toutes les zones vertes d'au moins min_area pixels (un objet peut apparaître
    en plusieurs morceaux) et renvoie (masque nettoyé, liste des contours gardés, aire totale)."""
    # ouverture : efface les petits points blancs isolés ; fermeture : bouche les petits trous dans la tache
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return clean, contours, float(sum(cv2.contourArea(c) for c in contours))


def zone_center(contours, shape):
    """Centre de masse (cx, cy) de l'ensemble des zones : moyenne des positions de tous leurs pixels verts.
    Un gros morceau pèse donc plus qu'un petit."""
    zone = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(zone, contours, -1, 255, cv2.FILLED)
    m = cv2.moments(zone, binaryImage=True)
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def main():
    cap = V4L2Camera(device=DEVICE, width=WIDTH, height=HEIGHT, fps=FPS)
    cap.open()
    if not cap.isOpened():
        print(f"Erreur: caméra inaccessible ({DEVICE}). Essayer DEVICE=/dev/video1 ou /dev/video2.", file=sys.stderr)
        sys.exit(1)

    win = f"Green detect {DEVICE}  |  Q=quitter"
    win_mask = "Masque vert (blanc = vert detecte)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_mask, 640, 480)
    for name, value in HSV_DEFAULTS.items():
        cv2.createTrackbar(name, win_mask, value, HSV_MAX[name], lambda x: None)
    cv2.createTrackbar("Aire min", win_mask, MIN_AREA_DEFAULT, 20000, lambda x: None)

    state = {"hsv": None}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None:
            apply_pipette(state["hsv"], x, y, win_mask)

    cv2.setMouseCallback(win, on_click)

    last_print = 0.0
    n = 0
    t_fps = time.time()
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Erreur: pas d'image reçue de {DEVICE}", file=sys.stderr)
            break

        n += 1
        elapsed = time.time() - t_fps
        if elapsed >= 1.0:
            fps_disp = n / elapsed
            n = 0
            t_fps = time.time()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        state["hsv"] = hsv
        t = {name: cv2.getTrackbarPos(name, win_mask) for name in HSV_DEFAULTS}
        lower = np.array([t["H min"], t["S min"], t["V min"]])
        upper = np.array([t["H max"], t["S max"], t["V max"]])
        mask = cv2.inRange(hsv, lower, upper)
        min_area = cv2.getTrackbarPos("Aire min", win_mask)
        mask, zones, area = find_green_zone(mask, min_area)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h}  {fps_disp:.1f} fps", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if zones:
            cv2.drawContours(frame, zones, -1, (0, 255, 255), 2)  # chaque morceau en jaune
            x, y, bw, bh = cv2.boundingRect(np.vstack(zones))  # un seul rectangle autour de l'ensemble
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.putText(frame, f"{len(zones)} morceau(x), aire {area:.0f} px", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cx, cy = zone_center(zones, frame.shape)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)  # contour blanc pour rester visible
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "VERT DETECTE", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"centre = ({cx}, {cy})", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if time.time() - last_print >= 0.5:  # 2 lignes/s max dans le terminal
                print(f"VERT DETECTE  centre = ({cx}, {cy})")
                last_print = time.time()
        else:
            cv2.putText(frame, "PAS DE VERT", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # repère du centre de l'image, pour voir l'écart avec le centre de la zone verte
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(win, frame)

        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        info = f"H {lower[0]}-{upper[0]}  S {lower[1]}-{upper[1]}  V {lower[2]}-{upper[2]}"
        cv2.putText(mask_view, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(win_mask, mask_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            print(f"seuils HSV : lower={lower.tolist()} upper={upper.tolist()}  aire min={min_area}")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
