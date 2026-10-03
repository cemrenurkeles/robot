"""Cinématique du follower SO-101 dans le repère de la TABLE, avec la correction mesurée sur le vrai robot.

Le modèle lerobot (placo + URDF SO-101) suppose que le 0° de chaque articulation est celui du fabricant ;
notre calibration diffère de quelques degrés. kinematics_correction.json contient ces décalages (mesurés en
comparant le modèle à des mesures à la règle) et la hauteur de la base au-dessus de la table (planche).

Repère : origine sur la table, sous l'axe de la base ; x vers l'avant, y vers la gauche, z vers le haut (cm).
Les angles manipulés ici sont ceux du robot (degrés lerobot, comme get_observation / send_action).
"""

import json
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "assets/so101/so101_new_calib.urdf"
CORRECTION_FILE = ROOT / "datasets/poses/kinematics_correction.json"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class ArmKinematics:
    def __init__(self, correction_file: Path = CORRECTION_FILE):
        self.kin = RobotKinematics(str(URDF), target_frame_name="gripper_frame_link", joint_names=ARM_JOINTS)
        # Les limites de l'URDF (ex. poignet ±95°) sont théoriques et décalées par notre correction : on les
        # désactive dans le solveur ; les vraies butées (calibration) sont vérifiées avant chaque mouvement.
        self.kin.solver.enable_joint_limits(False)
        corr = json.loads(correction_file.read_text()) if correction_file.exists() else {}
        self.offsets = {j: corr.get("offsets_deg", {}).get(j, 0.0) for j in ARM_JOINTS}
        self.base_height_cm = corr.get("base_height_cm", 0.0)

    def _model_angles(self, joints: dict) -> np.ndarray:
        """Angles robot {nom ou nom.pos: degrés} -> angles du modèle (correction appliquée)."""
        get = lambda j: joints[j] if j in joints else joints[f"{j}.pos"]  # noqa: E731
        return np.array([get(j) + self.offsets[j] for j in ARM_JOINTS])

    def pose_matrix(self, joints: dict) -> np.ndarray:
        """Matrice 4x4 de la pince (repère table, mètres)."""
        T = self.kin.forward_kinematics(self._model_angles(joints)).copy()
        T[2, 3] += self.base_height_cm / 100
        return T

    def gripper_cm(self, joints: dict) -> np.ndarray:
        """Position (x, y, z) de la pince en cm, repère table."""
        return self.pose_matrix(joints)[:3, 3] * 100

    def solve(self, joints: dict, target_cm, iterations: int = 300, tol_cm: float = 0.05) -> tuple[dict, float]:
        """Angles robot qui amènent la pince en target_cm (repère table) en gardant son orientation actuelle.
        Renvoie ({nom.pos: degrés} pour les 5 articulations du bras, erreur de position restante en cm)."""
        T_des = self.pose_matrix(joints)
        T_des[:3, 3] = np.asarray(target_cm, dtype=float) / 100
        T_des[2, 3] -= self.base_height_cm / 100  # repère table -> repère du modèle
        q = self._model_angles(joints)
        for _ in range(iterations):
            q = self.kin.inverse_kinematics(q, T_des, position_weight=1.0, orientation_weight=0.1)
            err = np.linalg.norm(self.kin.forward_kinematics(q)[:3, 3] - T_des[:3, 3]) * 100
            if err < tol_cm:
                break
        result = {f"{j}.pos": float(q[i] - self.offsets[j]) for i, j in enumerate(ARM_JOINTS)}
        return result, float(err)
