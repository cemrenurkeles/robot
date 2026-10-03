#!/usr/bin/env python3
"""Enregistre un mouvement fait avec le leader, puis le rejoue sur le follower autant de fois qu'on veut.

  record : tu bouges le leader, le follower suit en direct, le mouvement est sauvé dans un fichier JSON.
  export : extrait le mouvement (actions) d'un épisode d'un dataset LeRobot vers un fichier JSON.
  replay : le follower rejoint en douceur la pose de départ, puis rejoue le mouvement (--repeat, --speed).

Usage :
  python3 scripts/robot/motion.py record datasets/motions/salut.json --duration 10
  python3 scripts/robot/motion.py export cemrenurkeles/test_cube_all 0 datasets/motions/ep0.json
  python3 scripts/robot/motion.py replay datasets/motions/salut.json --repeat 3 --speed 0.5

Ports : FOLLOWER_PORT et LEADER_PORT (voir scripts/robot/identify_arms.py).
"""

import argparse
import json
import os
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

FPS = 30


def make_follower(hold: bool) -> SOFollower:
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0"),
            id=os.environ.get("FOLLOWER_ID", "follower_arm"),
            disable_torque_on_disconnect=not hold,
            max_relative_target=10.0,  # sécurité : jamais plus de 10° d'écart par commande
        )
    )
    robot.connect(calibrate=False)
    return robot


def move_to(robot: SOFollower, target: dict, duration: float) -> None:
    """Interpolation linéaire depuis la pose actuelle, pour éviter tout saut."""
    start = robot.get_observation()
    steps = max(1, int(duration * FPS))
    for i in range(1, steps + 1):
        a = i / steps
        robot.send_action({k: start[k] + (v - start[k]) * a for k, v in target.items()})
        time.sleep(1 / FPS)


def record(path: Path, duration: float) -> None:
    leader = SOLeader(
        SOLeaderTeleopConfig(
            port=os.environ.get("LEADER_PORT", "/dev/ttyACM1"), id=os.environ.get("LEADER_ID", "leader_arm")
        )
    )
    leader.connect(calibrate=False)
    robot = make_follower(hold=True)
    try:
        move_to(robot, leader.get_action(), 1.5)  # le follower rejoint d'abord le leader en douceur
        input(f"Prêt. Entrée pour démarrer l'enregistrement ({duration:.0f} s, Ctrl+C pour finir avant)...")
        frames, t0 = [], time.perf_counter()
        try:
            while (t := time.perf_counter() - t0) < duration:
                action = leader.get_action()
                robot.send_action(action)
                frames.append(action)
                print(f"\r  {t:4.1f} s  {len(frames)} poses", end="", flush=True)
                time.sleep(max(0.0, (len(frames) / FPS) - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
        print()
    finally:
        robot.disconnect()
        leader.disconnect()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fps": FPS, "frames": frames}))
    print(f"{len(frames)} poses ({len(frames) / FPS:.1f} s) sauvées dans {path}")


def export(repo_id: str, episode: int, path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, video_backend="pyav")
    if not 0 <= episode < ds.num_episodes:
        raise SystemExit(f"Épisode {episode} inexistant ({ds.num_episodes} épisodes, de 0 à {ds.num_episodes - 1})")
    names = ds.meta.features["action"]["names"]
    start = int(ds.meta.episodes["dataset_from_index"][episode])
    end = int(ds.meta.episodes["dataset_to_index"][episode])
    actions = ds.hf_dataset.select(range(start, end))["action"]
    frames = [{n: float(v) for n, v in zip(names, a)} for a in actions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fps": ds.fps, "frames": frames, "source": f"{repo_id} épisode {episode}"}))
    print(f"{repo_id} épisode {episode} : {len(frames)} poses ({len(frames) / ds.fps:.1f} s) -> {path}")


def replay(path: Path, repeat: int, speed: float, release: bool) -> None:
    data = json.loads(path.read_text())
    frames, fps = data["frames"], data["fps"]
    robot = make_follower(hold=not release)
    try:
        for n in range(1, repeat + 1):
            print(f"répétition {n}/{repeat} : retour à la pose de départ")
            move_to(robot, frames[0], 2.0)
            t0 = time.perf_counter()
            for i, pose in enumerate(frames):
                robot.send_action(pose)
                time.sleep(max(0.0, i / (fps * speed) - (time.perf_counter() - t0)))
        print("terminé")
    except KeyboardInterrupt:
        print("\narrêté")
    finally:
        robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("file", type=Path)
    r.add_argument("--duration", type=float, default=10.0)
    e = sub.add_parser("export")
    e.add_argument("repo_id")
    e.add_argument("episode", type=int)
    e.add_argument("file", type=Path)
    p = sub.add_parser("replay")
    p.add_argument("file", type=Path)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--speed", type=float, default=1.0, help="0.5 = deux fois plus lent (max 1.5)")
    p.add_argument("--release", action="store_true", help="couper le couple à la fin (le bras retombe)")
    args = parser.parse_args()

    if args.cmd == "record":
        record(args.file, args.duration)
    elif args.cmd == "export":
        export(args.repo_id, args.episode, args.file)
    else:
        if not 0.1 <= args.speed <= 1.5:
            raise SystemExit("--speed doit être entre 0.1 et 1.5")
        replay(args.file, args.repeat, args.speed, args.release)


if __name__ == "__main__":
    main()
