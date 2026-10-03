"""Ouvre un pipeline depthai minimal et vérifie qu'un frame RGB est bien capturé sur l'OAK-D Lite"""

import sys
import time

try:
    import depthai as dai
except ImportError:
    print("depthai not installed")
    sys.exit(1)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    queue = cam.requestOutput((300, 300)).createOutputQueue()
    pipeline.start()
    frame = queue.get()
    img = frame.getCvFrame()
    print(f"RGB frame: {img.shape} dtype={img.dtype}")
    print("OAK-D Lite OK")
    pipeline.stop()
    time.sleep(0.5)
