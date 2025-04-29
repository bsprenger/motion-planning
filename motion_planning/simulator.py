from motion_planning.environment import UltraTask
import numpy as np
import robosuite as suite
from typing import Any
from robosuite.utils.mjcf_utils import xml_path_completion

class Simulator:

    def __init__(self) -> None:
        self.env = self._make_env()

    @staticmethod
    def _make_env() -> Any:
        sim = suite.make(env_name="UltraTask")
        return sim

    def reset(self) -> None:
        self.env.reset()

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        observation = {}
        observation["robot0_joint_pos"] = obs["robot0_joint_pos"]
        observation["robot0_eef_pos"] = obs["robot0_eef_pos"]
        observation["robot0_eef_quat"] = obs["robot0_eef_quat"]
        observation["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"]
        observation["frontview_image"] = obs["frontview_image"]
        observation["frontview_depth"] = obs["frontview_depth"]
        return observation

    def render(self) -> None:
        self.env.render()

    @property
    def action_spec(self) -> Any:
        return self.env.action_spec
    def get_camera_transform(self, camera_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        camera_id = self.env.sim.model.camera_name2id("frontview")
        return self.env.sim.data.cam_xpos[camera_id], self.env.sim.data.cam_xmat[camera_id].reshape(3, 3)

    def get_robot_mjcf_path(self) -> str:
        return xml_path_completion("robots/panda/robot.xml")

    
