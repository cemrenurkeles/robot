"""Preview live d'une webcam USB générique (ET-S231 et similaires) via V4L2. Affiche résolution/fps réel à l'écran"""

import os
import sys
import time

import cv2

from camera_utils import V4L2Camera

DEVICE = os.environ.get("DEVICE", "/dev/video0")
WIDTH = int(os.environ.get("WIDTH", "1920"))
HEIGHT = int(os.environ.get("HEIGHT", "1080"))
FPS = int(os.environ.get("FPS", "30"))


def main():
    """Boucle d'affichage. Q/Echap pour quitter"""
    cap = V4L2Camera(device=DEVICE, width=WIDTH, height=HEIGHT, fps=FPS)
    cap.open()
    if not cap.isOpened():
        print(f"Erreur: caméra inaccessible ({DEVICE}).", file=sys.stderr)
        sys.exit(1)

    win = f"Viewer {DEVICE}  |  Q=quitter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    n = 0
    t_fps = time.time()
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: frame vide", file=sys.stderr)
            break

        n += 1
        elapsed = time.time() - t_fps
        if elapsed >= 1.0:
            fps_disp = n / elapsed
            n = 0
            t_fps = time.time()

        h, w = frame.shape[:2]
        info = f"{w}x{h}  {fps_disp:.1f} fps  (Q=quitter)"
        cv2.putText(frame, info, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
