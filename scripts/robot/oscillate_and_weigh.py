#!/usr/bin/env python3
"""Rejoue les deux épisodes oscillateur et pèse l'objet avec la balance à ressort (pesee_robot.py).

La caméra de scène (OAK) est ouverte avant le mouvement. L'oscillation commence à --debut secondes de l'épisode
--episode (temps de l'enregistrement ; par défaut 11 s de l'épisode 1 : le bras relâche le plateau entre 10,5 et
11 s). Le relevé de la balance démarre 1 s avant, en arrière-plan sans interrompre le mouvement, et dure au moins
pesee_robot.DUREE secondes et jusqu'à la fin de l'épisode + 3 s (ou --duree secondes) : pesee_robot retrouve
lui-même l'oscillation dans le relevé. À la fin des deux
épisodes, pesee_robot calcule la masse (fichiers positions.txt, mesure.csv, ajustement.png à côté de pesee_robot.py).

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/oscillate_and_weigh.py [--speed 0.5] [--episode 1] [--debut 11]

La caméra OAK ne doit pas être utilisée ailleurs (serveur oak_zmq_server.py arrêté).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion import make_follower, move_to  # noqa: E402
import pesee_robot  # noqa: E402

EPISODES = [Path("datasets/motions/oscillateur_ep0.json"), Path("datasets/motions/oscillateur_ep1.json")]
MARGE_S = 1.0  # le relevé démarre 1 s (temps de l'enregistrement) avant --debut, pour ne pas manquer le lâcher


class CameraOAKv3:
    """Même interface que pesee_robot.CameraOAK (lire, vider, fermer), pour DepthAI v3 installé dans ce projet.
    30 img/s : en v3, le capteur couleur de l'OAK-D Lite (IMX214) plafonne à 35 img/s (pesee_robot vise 60)."""

    def __init__(self, fps=30):
        import depthai as dai

        # juste après la fermeture d'un autre programme (ex. oak_zmq_server.py), l'OAK reste quelques secondes
        # « occupée » (X_LINK_BOOTED / INSUFFICIENT_PERMISSIONS) : on réessaie pendant 15 s
        deadline = time.time() + 15
        while True:
            try:
                self.pipeline = dai.Pipeline()
                break
            except RuntimeError as e:
                if time.time() > deadline:
                    raise RuntimeError(f"caméra OAK occupée (oak_zmq_server.py encore lancé ?) : {e}") from e
                print("caméra OAK encore occupée, nouvel essai...")
                time.sleep(2)
        cam = self.pipeline.create(dai.node.Camera).build()
        self.queue = cam.requestOutput((1280, 720), fps=fps).createOutputQueue(maxSize=8, blocking=False)
        self.pipeline.start()
        for _ in range(10):  # premières images (exposition pas encore stable)
            self.queue.get()

    def lire(self):
        """Renvoie (image BGR, instant de prise de vue en secondes, horloge de la caméra)."""
        img = self.queue.get()
        return img.getCvFrame(), img.getTimestamp().total_seconds()

    def vider(self):
        self.queue.tryGetAll()

    def fermer(self):
        self.pipeline.stop()


def play(robot, frames, fps, speed, on_time=None):
    """Rejoue les poses ; on_time(t) est appelé à chaque pose avec t = temps de l'enregistrement (s)."""
    t0 = time.perf_counter()
    for i, pose in enumerate(frames):
        if on_time is not None:
            on_time(i / fps)
        robot.send_action(pose)
        time.sleep(max(0.0, i / (fps * speed) - (time.perf_counter() - t0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=0.5, help="vitesse du rejeu (0.5 = deux fois plus lent)")
    parser.add_argument("--episode", type=int, default=1, choices=range(len(EPISODES)),
                        help="épisode où l'oscillation commence")
    parser.add_argument("--debut", type=float, default=11.0, help="début de l'oscillation, en s de cet épisode")
    parser.add_argument("--duree", type=float, default=None,
                        help="durée du relevé (s) ; par défaut au moins pesee_robot.DUREE et jusqu'à la fin + 3 s")
    args = parser.parse_args()
    if not 0.1 <= args.speed <= 1.5:
        raise SystemExit("--speed doit être entre 0.1 et 1.5")

    episodes = [json.loads(p.read_text()) for p in EPISODES]
    start_at = max(0.0, args.debut - MARGE_S)
    if args.duree is None:  # temps réel restant de l'épisode après le départ du relevé, + 3 s de marge
        rest = len(episodes[args.episode]["frames"]) / episodes[args.episode]["fps"] - start_at
        rest += sum(len(ep["frames"]) / ep["fps"] for ep in episodes[args.episode + 1:])
        args.duree = max(pesee_robot.DUREE + MARGE_S, rest / args.speed + 3.0)
    balance = pesee_robot.Balance(camera=CameraOAKv3())
    robot = make_follower(hold=True)
    started = False

    def start_weighing(t):
        nonlocal started
        if not started and t >= start_at:
            started = True
            balance.demarrer(args.duree)
            print(f"t = {t:.1f} s de l'épisode {args.episode} : relevé de la balance lancé ({args.duree:.0f} s)")

    try:
        for n, ep in enumerate(episodes):
            print(f"épisode {n} : retour à la pose de départ")
            move_to(robot, ep["frames"][0], 2.0)
            print(f"épisode {n} : rejeu ({len(ep['frames']) / ep['fps']:.1f} s à vitesse {args.speed})")
            play(robot, ep["frames"], ep["fps"], args.speed, start_weighing if n == args.episode else None)
        if not started:
            print(f"épisode {args.episode} plus court que {start_at} s : pas de relevé")
            return
        print("mouvement terminé, calcul de la masse...")
        try:
            r = balance.terminer()
            print(f"MASSE : {r['masse']:.2f}" + ("" if r["fiable"] else "  (peu fiable)"))
            print(f"graphe : {pesee_robot.FICHIER_GRAPHE}")
        except RuntimeError as e:
            print(f"mesure impossible : {e}")
    except KeyboardInterrupt:
        print("\narrêté")
    finally:
        robot.disconnect()
        balance.fermer()


if __name__ == "__main__":
    main()
