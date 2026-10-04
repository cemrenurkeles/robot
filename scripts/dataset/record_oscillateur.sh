#!/usr/bin/env bash
# Ajoute un épisode au dataset cemrenurkeles/oscillateur (reprise : les épisodes existants sont gardés).
# Le serveur de la caméra du dessus doit tourner dans un autre terminal :
#   OAK_WIDTH=1280 OAK_HEIGHT=720 python3 scripts/camera/oak_zmq_server.py
#
# Usage (depuis la racine du projet, venv activé) : bash scripts/dataset/record_oscillateur.sh [nombre d'épisodes, 1 par défaut]
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B7B0141411 \
  --robot.id=follower_arm \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B7B0152091 \
  --teleop.id=leader_arm \
  '--robot.cameras={ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: 1280, height: 720}, wrist: {type: opencv, index_or_path: 0, fps: 30, width: 640, height: 480} }' \
  --dataset.repo_id=cemrenurkeles/oscillateur \
  --dataset.root="$HOME/.cache/huggingface/lerobot/cemrenurkeles/oscillateur" \
  --dataset.single_task="oscillateur" \
  --dataset.num_episodes="${1:-1}" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30 \
  --resume=true \
  --dataset.push_to_hub=true \
  --dataset.private=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=false \
  --play_sounds=false
