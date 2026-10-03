"""Script de diagnostic : liste les caméras V4L2 disponibles (indices /dev/video*) et les devices OAK-D détectés via depthai"""

import cv2
import depthai as dai

MAX_INDEX = 10

print("Recherche V4L2 (indices 0-{})...".format(MAX_INDEX - 1))
found = []

for i in range(MAX_INDEX):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release()
        continue
    ret, _ = cap.read()
    if not ret:
        cap.release()
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    found.append((i, w, h, fps))
    print(f"  index={i}  {w}x{h}  {fps:.0f}fps  -> /dev/video{i}")

if not found:
    print("  Aucune caméra V4L2 détectée.")

print("\nRecherche OAK-D Lite (depthai)...")
try:
    devices = dai.Device.getAllConnectedDevices()
    if devices:
        for d in devices:
            print(f"  OAK: {d.name}  state={d.state}")
        print("  -> Utiliser make detect-oak pour tester un frame RGB")
    else:
        print("  Aucun device OAK détecté.")
except ImportError:
    print("  depthai non installé.")
