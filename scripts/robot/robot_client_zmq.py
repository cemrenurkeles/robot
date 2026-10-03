"""Wrapper autour de lerobot.async_inference.robot_client"""

from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.async_inference.robot_client import async_client, register_third_party_plugins

if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
