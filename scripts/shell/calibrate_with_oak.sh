#!/bin/bash
# Lance le serveur ZMQ OAK puis le script de calibration passé en argument
set -e

SCRIPT=$1
shift

python3 scripts/camera/oak_zmq_server.py &
ZMQ_PID=$!
trap "kill $ZMQ_PID 2>/dev/null || true" EXIT

echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
sleep 3

python3 scripts/camera/$SCRIPT "$@"
