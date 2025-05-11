import numpy as np


class TrajectoryController:
    """
    PI controller to control the end-effector position and orientation.
    The action output is a 7-DOF vector:
    [dx, dy, dz, droll, dpitch, dyaw, gripper_command].
    """

    def __init__(
        self,
        delta_translation_magnitude: float = 2.0,
        delta_rotation_magnitude: float = 0.1,  # Radians per step
        position_p_gain: float = 5.0,
        position_i_gain: float = 0.0,
        orientation_p_gain: float = 1.0,
        orientation_i_gain: float = 0.0,
        max_integral_pos: float = 0.5,  # Anti-windup limit for position
        max_integral_ori: float = 0.5,  # Anti-windup limit for orientation
    ):
        """
        Args:
            delta_translation_magnitude: The maximum magnitude of the translational part of
                                         the action (meters per step).
            delta_rotation_magnitude: The maximum magnitude of the rotational part (radians per step).
            position_p_gain: Proportional gain for position control.
            position_i_gain: Integral gain for position control.
            orientation_p_gain: Proportional gain for orientation control.
            orientation_i_gain: Integral gain for orientation control.
            max_integral_pos: Maximum allowed magnitude for position integral term (anti-windup).
            max_integral_ori: Maximum allowed magnitude for orientation integral term (anti-windup).
        """
        self.delta_translation_magnitude = delta_translation_magnitude
        self.delta_rotation_magnitude = delta_rotation_magnitude
        self.position_p_gain = position_p_gain
        self.position_i_gain = position_i_gain
        self.orientation_p_gain = orientation_p_gain
        self.orientation_i_gain = orientation_i_gain

        # Anti-windup limits
        self.max_integral_pos = max_integral_pos
        self.max_integral_ori = max_integral_ori

        # Integral terms
        self.position_integral = np.zeros(3)
        self.orientation_integral = np.zeros(3)

        # Previous references to detect changes
        self.prev_target_pos = None
        self.prev_target_orientation = None

    @staticmethod
    def rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
        """Converts Roll-Pitch-Yaw angles to a quaternion [x, y, z, w].
        Assumes ZYX extrinsic rotation convention (roll around x, then pitch around new y, then yaw around new z).
        """
        roll, pitch, yaw = rpy
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qx, qy, qz, qw])

    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        """Computes the inverse of a quaternion [x, y, z, w].
        For a unit quaternion, the inverse is its conjugate.
        """
        return np.array([-q[0], -q[1], -q[2], q[3]])

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """Multiplies two quaternions q1 * q0 ([x, y, z, w] format)."""
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        return np.array(
            [
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            ]
        )

    @staticmethod
    def quaternion_to_rpy(q: np.ndarray) -> np.ndarray:
        """Converts a quaternion [x, y, z, w] to Roll-Pitch-Yaw angles."""
        x, y, z, w = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _check_and_reset_targets(
        self, target_pos_abs: np.ndarray, target_orientation_rpy: np.ndarray
    ) -> None:
        """Check if targets have changed and reset integral terms if needed.

        Args:
            target_pos_abs: Target end-effector XYZ position
            target_orientation_rpy: Target end-effector orientation as RPY
        """
        if (
            self.prev_target_pos is None
            or not np.array_equal(target_pos_abs, self.prev_target_pos)
            or self.prev_target_orientation is None
            or not np.array_equal(target_orientation_rpy, self.prev_target_orientation)
        ):
            self.position_integral = np.zeros(3)
            self.orientation_integral = np.zeros(3)

        self.prev_target_pos = target_pos_abs.copy()
        self.prev_target_orientation = target_orientation_rpy.copy()

    def _compute_position_control(
        self, current_pos: np.ndarray, target_pos_abs: np.ndarray
    ) -> np.ndarray:
        """Compute position control action.

        Args:
            current_pos: Current end-effector position
            target_pos_abs: Target end-effector position

        Returns:
            Position control action (np.ndarray)
        """
        pos_error = target_pos_abs - current_pos
        self.position_integral += pos_error

        # Anti-windup
        position_integral_norm = np.linalg.norm(self.position_integral)
        if position_integral_norm > self.max_integral_pos:
            self.position_integral = (
                self.position_integral / position_integral_norm
            ) * self.max_integral_pos

        p_term_pos = self.position_p_gain * pos_error
        i_term_pos = self.position_i_gain * self.position_integral
        delta_pos_action = p_term_pos + i_term_pos

        # Output clipping
        pos_action_norm = np.linalg.norm(delta_pos_action)
        if pos_action_norm > self.delta_translation_magnitude:
            delta_pos_action = (
                delta_pos_action / pos_action_norm
            ) * self.delta_translation_magnitude

        return delta_pos_action

    def _compute_orientation_control(
        self,
        current_orientation_quat: np.ndarray,  # [x, y, z, w]
        target_orientation_rpy: np.ndarray,
    ) -> np.ndarray:
        """Compute orientation control action.

        Args:
            current_orientation_quat: Current orientation as quaternion [x,y,z,w]
            target_orientation_rpy: Target orientation as RPY angles

        Returns:
            Orientation control action [droll, dpitch, dyaw] (np.ndarray)
        """
        target_orientation_quat = self.rpy_to_quaternion(target_orientation_rpy)

        # TODO maybe move the scaling to the creation function
        current_orientation_quat_norm = np.linalg.norm(current_orientation_quat)
        if current_orientation_quat_norm < 1e-6:  # Avoid division by zero
            current_orientation_quat_normed = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            current_orientation_quat_normed = (
                current_orientation_quat / current_orientation_quat_norm
            )

        q_current_inv = self.quaternion_inverse(current_orientation_quat_normed)
        q_error = self.quaternion_multiply(target_orientation_quat, q_current_inv)

        # Use shortest path for q_error (since quaternions have double cover)
        if q_error[3] < 0:
            q_error = -q_error

        rpy_error = self.quaternion_to_rpy(q_error)
        # TODO maybe need to normalize?

        self.orientation_integral += rpy_error

        # Anti-windup
        orientation_integral_norm = np.linalg.norm(self.orientation_integral)
        if orientation_integral_norm > self.max_integral_ori:
            self.orientation_integral = (
                self.orientation_integral / orientation_integral_norm
            ) * self.max_integral_ori

        p_term_ori = self.orientation_p_gain * rpy_error
        i_term_ori = self.orientation_i_gain * self.orientation_integral
        delta_ori_action = p_term_ori + i_term_ori

        # Output clipping
        ori_action_norm = np.linalg.norm(delta_ori_action)
        if ori_action_norm > self.delta_rotation_magnitude:
            delta_ori_action = (
                delta_ori_action / ori_action_norm
            ) * self.delta_rotation_magnitude

        return delta_ori_action

    def compute_action(
        self,
        current_pos: np.ndarray,
        current_orientation_quat: np.ndarray,  # Expects [x, y, z, w]
        target_pos_abs: np.ndarray,
        target_orientation_rpy: np.ndarray,
        target_gripper_cmd: float,
    ) -> np.ndarray:
        """
        Computes the control action to move towards a target pose and achieve a gripper state.

        Args:
            current_pos: Current end-effector XYZ position (np.ndarray).
            current_orientation_quat: Current end-effector orientation as a quaternion [x, y, z, w] (np.ndarray). TODO make RPY?
            target_pos_abs: Target end-effector XYZ position (np.ndarray).
            target_orientation_rpy: Target end-effector orientation as RPY [roll, pitch, yaw] (np.ndarray).
            target_gripper_cmd: Target gripper command (float, -1 for open, 1 for close).

        Returns:
            action (np.ndarray): The computed 7-DOF action
                                   [dx, dy, dz, droll, dpitch, dyaw, gripper_command].
        """
        self._check_and_reset_targets(target_pos_abs, target_orientation_rpy)

        delta_pos_action = self._compute_position_control(current_pos, target_pos_abs)
        delta_ori_action = self._compute_orientation_control(
            current_orientation_quat, target_orientation_rpy
        )
        gripper_action_cmd = np.array([target_gripper_cmd])
        action = np.concatenate(
            (delta_pos_action, delta_ori_action, gripper_action_cmd)
        )

        return action
