"""Garde-fous moteurs : écriture position sans exception bloquante et config torque gripper obligatoire après connect()"""

def write_safe(bus, names, positions, normalize=False):
    """Écrit les positions sur plusieurs moteurs via sync_write"""
    goal_dict = {name: int(pos) for name, pos in zip(names, positions)}
    bus.sync_write("Goal_Position", goal_dict, normalize=normalize)


def configure_gripper(bus):
    """Limite torque gripper. Copie de SOFollower.configure() dans lerobot"""
    bus.write("Max_Torque_Limit", "gripper", 500)
    bus.write("Protection_Current", "gripper", 250)
    bus.write("Overload_Torque", "gripper", 25)
