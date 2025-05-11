import numpy as np


class WaypointPlanner:
    """Uses motion primitives to generate waypoints for the robot."""

    def __init__(self, gripper_open_cmd: float = -1.0, gripper_close_cmd: float = 1.0):
        """
        Args:
            gripper_open_cmd (float): Command value for an open gripper.
            gripper_close_cmd (float): Command value for a closed gripper.
        """
        self.gripper_open_cmd = gripper_open_cmd
        self.gripper_close_cmd = gripper_close_cmd

    def _move_to(
        self,
        target_pos: np.ndarray,
        gripper_cmd: float,
        label_prefix: str = "Move to",
    ) -> dict:
        """
        Generates a waypoint to move the end-effector to a target position
        with a specified gripper command.

        Args:
            target_pos (np.ndarray): The target end-effector position.
            gripper_cmd (float): The command for the gripper.
            label_prefix (str): Prefix for the waypoint label.

        Returns:
            dict: A waypoint dictionary for the TaskExecutor.
        """
        return {
            "label": f"{label_prefix} Position",
            "target_pos_abs": target_pos,
            "target_gripper_cmd": gripper_cmd,
        }

    def _open_gripper(self, pos: np.ndarray) -> dict:
        """
        Args:
            pos (np.ndarray): The end-effector position where the gripper opens.

        Returns:
            dict: A waypoint dictionary for the TaskExecutor.
        """
        return {
            "label": "Open Gripper",
            "target_pos_abs": pos,
            "target_gripper_cmd": self.gripper_open_cmd,
            "type": "gripper_actuation",
        }

    def _close_gripper(self, pos: np.ndarray) -> dict:
        """
        Args:
            pos (np.ndarray): The end-effector position where the gripper closes.

        Returns:
            dict: A waypoint dictionary for the TaskExecutor.
        """
        return {
            "label": "Close Gripper",
            "target_pos_abs": pos,
            "target_gripper_cmd": self.gripper_close_cmd,
            "type": "gripper_actuation",
        }

    def plan_pick_trajectory(
        self,
        current_location: np.ndarray,
        pick_location: np.ndarray,
        height_offset: float = 0.1,
    ) -> list[dict]:
        """
        Generates a sequence of waypoints for a pick operation, ending with gripper closed.
        The sequence involves:
        1. Moving directly upwards from current position to a safe height offset.
        2. Moving horizontally to be above the pick location at the safe height.
        3. Moving down to the actual pick location.
        4. Closing the gripper to grasp the object.

        Args:
            current_location (np.ndarray): The starting position from which to plan the trajectory.
            pick_location (np.ndarray): The 3D coordinate for grasping the object.
            safe_height_offset (float): How much higher than pick_location Z to position for approach.

        Returns:
            list[dict]: A list of waypoint dictionaries for the TaskExecutor.
        """
        waypoints = []
        safe_height = current_location[2] + height_offset

        # 1. Move directly upwards to a safe height with gripper open
        up_position = np.array([current_location[0], current_location[1], safe_height])
        waypoints.append(
            self._move_to(
                up_position,
                self.gripper_open_cmd,
                label_prefix="Move Up to Safe Height",
            )
        )

        # 2. Move horizontally to be above the pick location at the safe height
        above_pick = np.array([pick_location[0], pick_location[1], safe_height])
        waypoints.append(
            self._move_to(
                above_pick,
                self.gripper_open_cmd,
                label_prefix="Move Above Pick Position",
            )
        )

        # 3. Move down to the actual pick_location with gripper open
        waypoints.append(
            self._move_to(
                pick_location,
                self.gripper_open_cmd,
                label_prefix="Move to Pick Position",
            )
        )

        # 4. Close the gripper at pick_location
        waypoints.append(self._close_gripper(pick_location))

        return waypoints

    def plan_place_trajectory(
        self,
        current_location: np.ndarray,
        release_location: np.ndarray,
        height_offset: float = 0.1,
    ) -> list[dict]:
        """
        Generates a sequence of waypoints for a place operation, ending with gripper open.
        The sequence involves:
        1. Moving directly upwards to a safe height offset from release location.
        2. Moving horizontally to be above the release location at the safe height.
        3. Moving down to the actual release location.
        4. Opening the gripper to release the object.

        Args:
            approach_location (np.ndarray): The starting position from which to plan the trajectory.
            release_location (np.ndarray): The 3D coordinate where the gripper opens to release the object.
            safe_height_offset (float): How much higher than release_location Z to position for approach.

        Returns:
            list[dict]: A list of waypoint dictionaries for the TaskExecutor.
        """
        waypoints = []

        # Calculate the safe height based on the release location
        safe_height = release_location[2] + height_offset

        # 1. Move directly upwards to a safe height with gripper closed
        up_position = np.array([current_location[0], current_location[1], safe_height])
        waypoints.append(
            self._move_to(
                up_position,
                self.gripper_close_cmd,
                label_prefix="Move Up to Safe Height",
            )
        )

        # 2. Move horizontally to be above the release location at the safe height
        above_release = np.array(
            [release_location[0], release_location[1], safe_height]
        )
        waypoints.append(
            self._move_to(
                above_release,
                self.gripper_close_cmd,
                label_prefix="Move Above Release Position",
            )
        )

        # 3. Move down to the actual release_location with gripper still closed
        waypoints.append(
            self._move_to(
                release_location,
                self.gripper_close_cmd,
                label_prefix="Move to Release Position",
            )
        )

        # 4. Open the gripper at release_location
        waypoints.append(self._open_gripper(release_location))

        return waypoints
