#!/usr/bin/env python3
"""Mémorise une pose du follower, choisie à la main avec le leader (le follower suit le leader en direct).

Quand la pose convient, Entrée (ou `kill -USR1 <pid>` depuis un autre terminal) : les 6 positions (degrés)
sont écrites dans un fichier JSON.
Le robot ne bouge que parce que tu bouges le leader.

Usage :
  FOLLOWER_PORT=/dev/cu.usbmodemXXXX LEADER_PORT=/dev/cu.usbmodemYYYY \
    python3 scripts/robot/save_pose.py [datasets/poses/observation.json]
"""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

FPS = 30


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/poses/observation.json")
    leader = SOLeader(
        SOLeaderTeleopConfig(
            port=os.environ.get("LEADER_PORT", "/dev/ttyACM1"), id=os.environ.get("LEADER_ID", "leader_arm")
        )
    )
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
            id=os.environ.get("FOLLOWER_ID", "follower_arm"),
            disable_torque_on_disconnect=False,  # le bras reste dans la pose enregistrée
            max_relative_target=10.0,
        )
    )
    leader.connect(calibrate=False)
    robot.connect(calibrate=False)

    done = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: done.set())
    if sys.stdin.isatty():
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    print(f"PID {os.getpid()}", flush=True)
    print("Place le bras avec le leader (caméra du poignet vers le bas), puis Entrée pour enregistrer la pose.")
    try:
        while not done.is_set():
            robot.send_action(leader.get_action())
            time.sleep(1 / FPS)
        time.sleep(0.5)  # laisse le follower finir sa course
        pose = robot.get_observation()
    finally:
        robot.disconnect()
        leader.disconnect()

    pose = {k: round(float(v), 2) for k, v in pose.items() if k.endswith(".pos")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pose, indent=2))
    print(f"Pose enregistrée dans {path} :")
    for k, v in pose.items():
        print(f"  {k:18} {v:7.1f}°")


if __name__ == "__main__":
    main()
