#!/usr/bin/env bash
# Attrape l'objet vert (find_green.py, inchangé), puis rejoue les deux épisodes du dataset oscillateur en pesant
# l'objet avec la balance (oscillate_and_weigh.py : caméra de scène OAK, relevé à partir de la 11e seconde).
# Le rejeu ne démarre que si find_green.py a bien pris l'objet et est revenu à la pose d'observation.
#
# Usage (depuis la racine du projet, venv activé, serveur oak_zmq_server.py arrêté) :
#   FOLLOWER_PORT=/dev/cu.usbmodemXXXX [DEVICE=/dev/video0] [SPEED=0.5] [SPEED2=1.5] [HAUTEUR=3.0] [DEBUT=11] \
#     bash scripts/robot/grab_and_oscillate.sh
set -o pipefail

LOG=$(mktemp)
# HAUTEUR : hauteur du bout de la pince au-dessus de la table en fin de descente (find_green : 3.5 par défaut)
python3 -u scripts/robot/find_green.py --hauteur "${HAUTEUR:-3.0}" "$@" 2>&1 | tee "$LOG"

if grep -q "RETOUR à la pose d'observation" "$LOG"; then
    echo "objet pris : rejeu des épisodes oscillateur 0 et 1 avec pesée"
    python3 scripts/robot/oscillate_and_weigh.py --speed "${SPEED:-0.5}" --speed2 "${SPEED2:-${SPEED:-0.5}}" --debut "${DEBUT:-11}"
else
    echo "objet non pris : pas de rejeu"
    exit 1
fi
