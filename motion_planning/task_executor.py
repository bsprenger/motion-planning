import math

import numpy as np

from motion_planning.simulator import Simulator
from motion_planning.trajectory_controller import TrajectoryController
from motion_planning.waypoint_planner import WaypointPlanner


class TaskExecutor:
    """
    Executes a sequence of waypoints and higher-level tasks like pick and place.
    """

    def __init__(
        self,
        simulator: Simulator,
        controller: TrajectoryController,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.05,  # Radians, approx 3 degrees
    ):
        """
        Args:
            simulator: The simulation environment instance (e.g., from Simulator class).
                       Used for stepping and rendering.
            controller: The TrajectoryController instance for computing actions.
            position_tolerance: The maximum allowed distance (meters) to the target position
                                for it to be considered "reached".
            orientation_tolerance: The maximum allowed angular difference (radians) to the target
                                   orientation for it to be considered "reached".
        """
        self.sim = simulator
        self.controller = controller
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.waypoint_planner = WaypointPlanner()

    def _check_pose_reached(
        self,
        current_pos: np.ndarray,
        current_orientation_quat: np.ndarray,  # [x,y,z,w]
        target_pos_abs: np.ndarray,
        target_orientation_rpy: np.ndarray,
    ) -> tuple[bool, bool]:
        """
        Checks if the target pose (position and orientation) has been reached.

        Args:
            current_pos: Current end-effector XYZ position.
            current_orientation_quat: Current end-effector orientation as quaternion [x,y,z,w].
            target_pos_abs: Target end-effector XYZ position.
            target_orientation_rpy: Target end-effector orientation as RPY.

        Returns:
            A tuple (position_reached, orientation_reached).
        """
        pos_error = target_pos_abs - current_pos
        dist_to_target_pos = np.linalg.norm(pos_error)
        position_reached = dist_to_target_pos < self.position_tolerance

        target_orientation_quat = self.controller.rpy_to_quaternion(
            target_orientation_rpy
        )

        # Ensure current_orientation_quat is unit
        current_orientation_quat_norm = np.linalg.norm(current_orientation_quat)
        if current_orientation_quat_norm < 1e-6:  # Avoid division by zero
            current_orientation_quat_normed = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            current_orientation_quat_normed = (
                current_orientation_quat / current_orientation_quat_norm
            )

        # q_error represents rotation from current to target: q_target = q_error * q_current
        # So, q_error = q_target * q_current_inverse
        q_current_inv = self.controller.quaternion_inverse(
            current_orientation_quat_normed
        )
        q_error = self.controller.quaternion_multiply(
            target_orientation_quat, q_current_inv
        )

        # Ensure shortest path for q_error (w component should be non-negative)
        if q_error[3] < 0:
            q_error = -q_error

        rpy_error = self.controller.quaternion_to_rpy(q_error)
        angular_dist_to_target = np.linalg.norm(rpy_error)
        orientation_reached = angular_dist_to_target < self.orientation_tolerance

        return position_reached, orientation_reached

    def reset_pose(
        self, initial_obs: dict, max_steps: int = 200, gripper_steps: int = 10
    ) -> tuple[dict, bool]:
        """
        Resets the robot EE to a predefined home pose (XYZ: [0,0,1], RPY: [pi,0,0]) with an open gripper.

        Args:
            initial_obs: The initial observation dictionary from the simulator.
            max_steps: Maximum simulation steps to attempt for this reset operation.
            gripper_steps: Number of steps to hold the gripper action.

        Returns:
            A tuple containing:
                - final_obs (dict): The final observation dictionary after the reset.
                - success (bool): True if the reset pose was reached, False otherwise.
        """
        reset_waypoint = {
            "target_pos_abs": np.array([0.0, 0.0, 0.90]),
            "target_orientation_rpy": np.array([math.pi, 0.0, 0.0]),
            "target_gripper_cmd": -1.0,
            "label": "Reset Robot Pose to Home",
            "type": "reset_action",
        }
        return self.execute_waypoint_sequence(
            waypoints=[reset_waypoint],
            initial_obs=initial_obs,
            max_steps_per_waypoint=max_steps,
            gripper_action_steps=gripper_steps,
        )

    def execute_waypoint_sequence(
        self,
        waypoints: list[dict],
        initial_obs: dict,
        max_steps_per_waypoint: int = 200,
        gripper_action_steps: int = 20,
    ) -> tuple[dict, bool]:
        """
        Executes a given sequence of waypoints.

        Args:
            waypoints: A list of waypoint dictionaries. Each dictionary should contain:
                - 'target_pos_abs' (np.ndarray): Target XYZ position.
                - 'target_gripper_cmd' (float): Target gripper command (-1 open, 1 closed).
                - 'label' (str, optional): A label for the waypoint for logging.
                - 'target_orientation_rpy' (np.ndarray, optional): Target RPY orientation.
                                                                    Defaults to [0,0,0] if not provided.
            initial_obs: The initial observation dictionary from the simulator.
                         Expected to contain 'robot0_eef_pos' and 'robot0_eef_quat'.
            max_steps_per_waypoint: Maximum simulation steps to attempt for each waypoint.
            gripper_action_steps: Number of steps to hold a gripper action (e.g., close/open) to ensure completion.

        Returns:
            A tuple containing:
                - final_obs (dict): The final observation dictionary after executing all waypoints.
                - success (bool): True if all waypoints were reached (pose and gripper) within their step limits, False otherwise.
        """
        current_obs = initial_obs
        overall_success = True

        for i, waypoint in enumerate(waypoints):
            target_pos_abs = waypoint["target_pos_abs"]
            target_gripper_cmd = waypoint["target_gripper_cmd"]
            label = waypoint.get("label", f"Waypoint {i + 1}")
            # Default to no rotation if not specified
            target_orientation_rpy = waypoint.get(
                "target_orientation_rpy", np.array([math.pi, 0.0, 0.0])
            )

            print(
                f"\nExecuting: {label} -> Target Pos: {np.round(target_pos_abs, 3)}, "
                f"Target RPY: {np.round(target_orientation_rpy, 3)}, Gripper: {target_gripper_cmd}"
            )

            pose_reached_for_waypoint = False
            for step_num in range(max_steps_per_waypoint):
                current_pos = current_obs["robot0_eef_pos"]
                current_quat = current_obs["robot0_eef_quat"]

                action = self.controller.compute_action(
                    current_pos=current_pos,
                    current_orientation_quat=current_quat,
                    target_pos_abs=target_pos_abs,
                    target_orientation_rpy=target_orientation_rpy,
                    target_gripper_cmd=target_gripper_cmd,
                )

                current_obs = self.sim.step(action)
                self.sim.render()

                # Check if pose is reached after the step
                updated_current_pos = current_obs["robot0_eef_pos"]
                updated_current_quat = current_obs["robot0_eef_quat"]
                position_reached, orientation_reached = self._check_pose_reached(
                    updated_current_pos,
                    updated_current_quat,
                    target_pos_abs,
                    target_orientation_rpy,
                )
                pose_reached_this_step = position_reached and orientation_reached

                is_gripper_waypoint = waypoint.get("type") == "gripper_actuation"

                if is_gripper_waypoint and step_num < gripper_action_steps - 1:
                    if (step_num + 1) % 5 == 0:
                        print(
                            f"  Holding gripper action for {label} ({step_num + 1}/{gripper_action_steps} steps)"
                        )
                    continue
                elif is_gripper_waypoint and step_num >= gripper_action_steps - 1:
                    print(
                        f"  Gripper action {label} likely complete after {gripper_action_steps} steps."
                    )
                    pose_reached_for_waypoint = True
                    break

                if pose_reached_this_step and not is_gripper_waypoint:
                    print(f"  Reached target pose for {label} in {step_num + 1} steps.")
                    pose_reached_for_waypoint = True
                    break

                if (step_num + 1) % 25 == 0 and not is_gripper_waypoint:
                    dist_to_target = np.linalg.norm(target_pos_abs - current_pos)
                    print(
                        f"  {label} - Step {step_num + 1}: Dist to Target Pos={dist_to_target:.4f} m"
                    )

            if not pose_reached_for_waypoint:
                print(
                    f"  Warning: Max steps ({max_steps_per_waypoint}) reached for {label}."
                )
                overall_success = False

        if overall_success:
            print("\nSuccessfully executed all waypoints.")
        else:
            print("\nWaypoint sequence execution completed with warnings or failures.")

        return current_obs, overall_success

    def perform_pick_and_place(
        self,
        initial_obs: dict,
        pick_location: np.ndarray,
        target_location: np.ndarray,
        pick_approach_offset: float = 0.1,
        object_release_clearance: float = 0.02,
    ) -> dict:
        """
        Performs a single pick and place operation.

        Args:
            initial_obs: The initial observation dictionary from the simulator.
            pick_location: The 3D coordinate for grasping the object.
            target_location: The 3D coordinate for placing the object.
            pick_approach_offset: Height offset for the approach to the pick location.
            object_release_clearance: Clearance height for the end-effector at release.

        Returns:
            The final observation dictionary after the place operation.
        """
        obs = initial_obs

        # 1. Plan and execute PICK trajectory (ends with gripper closed at pick_location)
        print(f"Starting PICK operation from: {pick_location}")
        pick_waypoints = self.waypoint_planner.plan_pick_trajectory(
            current_location=obs["robot0_eef_pos"],
            pick_location=pick_location,
            height_offset=pick_approach_offset,
        )
        obs, _ = self.execute_waypoint_sequence(
            waypoints=pick_waypoints,
            initial_obs=obs,
        )
        print(f"Pick (grasp) completed. Current EEF pos: {obs['robot0_eef_pos']}")

        # 2. Plan and execute PLACE trajectory (approaches, moves to release, opens gripper)
        # target_location[2] now contains the Z for the base of the object being placed.
        # object_release_clearance is added to this for the gripper's release height.
        actual_release_location = np.array(
            [
                target_location[0],
                target_location[1],
                target_location[2] + object_release_clearance,
            ]
        )

        print(
            f"Starting PLACE operation. Current pos: {obs['robot0_eef_pos']}, Release at: {actual_release_location}"
        )
        place_waypoints = self.waypoint_planner.plan_place_trajectory(
            current_location=obs["robot0_eef_pos"],
            release_location=actual_release_location,
            height_offset=0,  # Height offset for place is handled by actual_release_location
        )
        obs, _ = self.execute_waypoint_sequence(
            waypoints=place_waypoints,
            initial_obs=obs,
        )
        print(f"Place (release) completed. Current EEF pos: {obs['robot0_eef_pos']}")

        return obs
