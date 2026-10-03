"""Capture une photo unique depuis l'OAK-D Lite (avec white balance + crop optionnel) et la sauvegarde en JPEG"""

import os
import sys
from datetime import datetime
import cv2

from camera_utils import white_balance, OAKCamera

output = os.environ.get("OUTPUT_FILE", f"/workspace/datasets/photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")

with OAKCamera() as cap:
    if not cap.isOpened():
        print("Erreur: OAK-D Lite inaccessible.", file=sys.stderr)
        sys.exit(1)
    ret, frame = cap.read()

if not ret or frame is None:
    print("Erreur: frame vide", file=sys.stderr)
    sys.exit(1)

frame = white_balance(frame)

crop_x = int(os.environ.get("CROP_X", "0"))
crop_y = int(os.environ.get("CROP_Y", "0"))
crop_w = int(os.environ.get("CROP_W", "0"))
crop_h = int(os.environ.get("CROP_H", "0"))
# Crop optionnel, sert notamment à retirer le bras leader/main du champ
if crop_w > 0 and crop_h > 0:
    frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    print(f"Crop ROI: x={crop_x} y={crop_y} w={crop_w} h={crop_h}")

outdir = os.path.dirname(output)
if outdir:
    os.makedirs(outdir, exist_ok=True)
cv2.imwrite(output, frame)
print(f"Photo sauvegardée : {output}")
