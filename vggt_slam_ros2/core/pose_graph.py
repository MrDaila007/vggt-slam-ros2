"""
Pose graph optimisation for loop closure (Stage 2.2).

Uses a GTSAM NonlinearFactorGraph with:
  - PriorFactorPose3  on the first pose (fixes gauge freedom)
  - BetweenFactorPose3 for consecutive window poses (odometry)
  - BetweenFactorPose3 for matched frame pairs (loop closure)

Graph is optimised with Levenberg-Marquardt after each loop closure.

Pose convention (throughout this module):
  T_world_cam : (4, 4) SE(3) with R = rotation, t = camera position
  extrinsic   : (3, 4) cam-from-world  [R | -R @ t_world]

  T_world_cam = inv(extrinsic_to_4x4)

GTSAM is an optional dependency; an ImportError is raised only when
`PoseGraph` methods are actually called.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

_GTSAM_AVAILABLE = False
try:
    import gtsam
    _GTSAM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def extrinsic_to_world(ext: np.ndarray) -> np.ndarray:
    """
    (3, 4) cam-from-world  →  (4, 4) world-from-cam SE(3).
    """
    R = ext[:3, :3]
    t = ext[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def world_to_extrinsic(T: np.ndarray) -> np.ndarray:
    """
    (4, 4) world-from-cam SE(3)  →  (3, 4) cam-from-world extrinsic.
    """
    R_w2c = T[:3, :3].T
    t_w2c = -R_w2c @ T[:3, 3]
    ext = np.zeros((3, 4), dtype=np.float64)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = t_w2c
    return ext


def relative_pose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    SE(3) relative pose from frame A to frame B: T_a_b = inv(T_a) @ T_b.
    Both inputs are (4, 4) world-from-cam.
    """
    return np.linalg.inv(T_a) @ T_b


# ---------------------------------------------------------------------------
# GTSAM conversion helpers
# ---------------------------------------------------------------------------

def _np_to_gtsam_pose3(T: np.ndarray) -> "gtsam.Pose3":
    """(4,4) SE(3) → gtsam.Pose3."""
    rot = gtsam.Rot3(T[:3, :3])
    pt = gtsam.Point3(T[0, 3], T[1, 3], T[2, 3])
    return gtsam.Pose3(rot, pt)


def _gtsam_pose3_to_np(pose: "gtsam.Pose3") -> np.ndarray:
    """gtsam.Pose3 → (4,4) SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


# ---------------------------------------------------------------------------
# PoseGraph
# ---------------------------------------------------------------------------

class PoseGraph:
    """
    Incremental pose graph for SLAM back-end optimisation.

    Usage
    -----
    pg = PoseGraph()
    pg.add_pose(T_world_cam_0)            # first pose → adds prior
    pg.add_pose(T_world_cam_1)            # subsequent → adds odometry
    pg.add_loop(from_idx=0, T_rel=T_0_5) # loop closure
    corrected = pg.optimize()             # {idx: (4,4) T_world_cam}
    """

    # Noise models (sigmas in [rad, rad, rad, m, m, m])
    _ODOM_SIGMAS    = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    _PRIOR_SIGMAS   = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    _LOOP_SIGMAS    = np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10])

    def __init__(self) -> None:
        self._check_gtsam()
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._poses: list[np.ndarray] = []   # (4,4) world-from-cam

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_pose(self, T_world_cam: np.ndarray) -> int:
        """
        Add a new pose to the graph.

        For the first pose, a PriorFactorPose3 is added.
        For subsequent poses, a BetweenFactorPose3 (odometry) is added.

        Returns the index of the new pose.
        """
        idx = len(self._poses)
        key = gtsam.symbol('x', idx)

        pose_g = _np_to_gtsam_pose3(T_world_cam)
        self._initial.insert(key, pose_g)
        self._poses.append(T_world_cam.copy())

        if idx == 0:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self._PRIOR_SIGMAS)
            self._graph.add(gtsam.PriorFactorPose3(key, pose_g, prior_noise))
        else:
            prev_key = gtsam.symbol('x', idx - 1)
            T_rel = relative_pose(self._poses[idx - 1], T_world_cam)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(self._ODOM_SIGMAS)
            rel_g = _np_to_gtsam_pose3(T_rel)
            self._graph.add(
                gtsam.BetweenFactorPose3(prev_key, key, rel_g, odom_noise)
            )

        return idx

    def add_loop(
        self,
        from_idx: int,
        to_idx: int,
        T_rel: np.ndarray,
    ) -> None:
        """
        Add a loop closure factor.

        T_rel : (4,4) SE(3) — relative pose from `from_idx` to `to_idx`,
                measured independently (e.g. from VGGT re-inference).
        """
        key_a = gtsam.symbol('x', from_idx)
        key_b = gtsam.symbol('x', to_idx)
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(self._LOOP_SIGMAS)
        self._graph.add(
            gtsam.BetweenFactorPose3(key_a, key_b, _np_to_gtsam_pose3(T_rel), loop_noise)
        )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self) -> dict[int, np.ndarray]:
        """
        Run Levenberg-Marquardt optimisation.

        Returns
        -------
        dict mapping pose index → (4, 4) corrected world-from-cam SE(3).
        """
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self._graph, self._initial, params
        )
        result = optimizer.optimize()

        corrected: dict[int, np.ndarray] = {}
        for i in range(len(self._poses)):
            key = gtsam.symbol('x', i)
            pose_g = result.atPose3(key)
            corrected[i] = _gtsam_pose3_to_np(pose_g)
        return corrected

    def update_initial_values(self, corrected: dict[int, np.ndarray]) -> None:
        """Update stored initial values to the optimised result (warm-start)."""
        for i, T in corrected.items():
            self._poses[i] = T
            key = gtsam.symbol('x', i)
            if self._initial.exists(key):
                self._initial.update(key, _np_to_gtsam_pose3(T))

    # ------------------------------------------------------------------

    @property
    def pose_count(self) -> int:
        return len(self._poses)

    @staticmethod
    def _check_gtsam() -> None:
        if not _GTSAM_AVAILABLE:
            raise RuntimeError(
                "gtsam is not installed. "
                "Install it with: pip install gtsam"
            )
