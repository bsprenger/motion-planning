"""
Sources:
- https://en.wikipedia.org/wiki/Camera_matrix
- https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/PinholeCamera/PinholeCamera.html
- https://ksimek.github.io/2013/08/13/intrinsic/
- https://mujoco.readthedocs.io/en/3.0.1/modeling.html#cameras for coordinate system
"""

import cv2
import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from .simulator import Simulator
from .utils import VALID_COLOR_CHARS


def create_color_mask(image: np.ndarray, color: VALID_COLOR_CHARS) -> np.ndarray:
    """
    Create a binary mask for a specified color in an image.

    Args:
        image: The input RGB image as a NumPy array.
        color: The color to segment ("r", "g", or "b").

    Returns:
        A binary mask where the specified color is white (255) and all other colors are black (0).
        TODO: check 255 or 1

    Raises:
        ValueError: If an invalid color is provided.
    """
    # We convert to HSV for more intuitive color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # TODO: parameterize the color ranges
    if color == "r":
        # Red can span across 0-10 and 170-180 in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "g":
        lower_green = np.array([36, 50, 50])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
    elif color == "b":
        lower_blue = np.array([100, 150, 0])  # Adjusted lower bound for blue
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    else:
        raise ValueError(f"Invalid color: {color}. Valid options are 'r', 'g', 'b'.")

    kernel = np.ones((5, 5), np.uint8)  # TODO: parameterize kernel size

    # Apply morphological operations to clean up the mask
    # Morphological opening removes noise (erodes then dilates)
    # Morphological closing fills small holes (dilates then erodes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


class CameraProcessor:
    """
    Extracts pixel positions in the world frame based on color segmentation,
    camera intrinsics/extrinsics, and depth information.
    """

    def __init__(self, simulator: Simulator, camera_name: str = "frontview") -> None:
        self.camera_intrinsics = simulator.get_camera_intrinsics()
        self.t, self.R = simulator.get_camera_transform()
        self.camera_name = camera_name
        self.sim = simulator.env.sim

    def get_pixel_coords_from_color(
        self, observation: dict[str, np.ndarray], color: VALID_COLOR_CHARS
    ) -> np.ndarray:
        """
        Extract pixel coordinates corresponding to a specific color in the image.

        Args:
            observation: Dictionary containing camera observations.
            color: Color to detect in the image.

        Returns:
            2D array of pixel coordinates with shape (2, N) where N is the number of detected pixels.
            The first row contains x-coordinates, the second row contains y-coordinates.
        """
        rgb_image = observation[f"{self.camera_name}_image"]
        bitmask = create_color_mask(rgb_image, color)

        y, x = np.where(bitmask)

        # If no pixels match the color, return empty array with correct shape
        if len(x) == 0:
            return np.zeros((2, 0), dtype=np.int32)

        pixel_coords_2d = np.vstack((x, y))
        return pixel_coords_2d

    def get_world_frame_positions_from_color(
        self, observation: dict[str, np.ndarray], color: VALID_COLOR_CHARS
    ) -> np.ndarray:
        """
        Convert color-detected pixels to 3D world coordinates.

        Args:
            observation: Dictionary containing camera observations.
            color: Color to detect and convert to 3D positions.

        Returns:
            3D points in the world frame as a (3, N) array where N is the number of detected points.

        Raises:
            KeyError: If required observation keys are missing.
            ValueError: If no pixels of the specified color are detected.
        """
        pixel_coords = self.get_pixel_coords_from_color(observation, color)

        # If no pixels were detected, return empty array with correct shape
        if pixel_coords.shape[1] == 0:
            return np.zeros((3, 0), dtype=np.float32)

        depth_image = observation[f"{self.camera_name}_depth"]
        depth_image_unnormalized = get_real_depth_map(self.sim, depth_image)
        depth_values = depth_image_unnormalized[pixel_coords[1], pixel_coords[0]]

        points_in_camera_frame = self._unproject_points_to_camera_frame(
            pixel_coords, depth_values[:, 0]
        )

        return self._camera_to_world_frame(points_in_camera_frame)

    def get_block_position_from_color(
        self, observation: dict[str, np.ndarray], color: VALID_COLOR_CHARS
    ) -> np.ndarray:
        """
        Get the average position of a block of a specified color in the world frame.

        Args:
            observation: Dictionary containing camera observations.
            color: Color of the block to detect.

        Returns:
            Average 3D position of the detected block in the world frame.
        """
        positions = self.get_world_frame_positions_from_color(observation, color)
        return np.mean(positions, axis=1)

    def _unproject_points_to_camera_frame(
        self, pixel_coords_2d: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """
        Convert pixel coordinates and depth values to 3D points in the camera frame.

        Args:
            pixel_coords_2d: Pixel coordinates as a (2, N) array (x, y).
            depth: Depth values for each pixel as a (N,) array.

        Returns:
            3D points in the camera frame as a (3, N) array.
        """
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        # TODO switch from x,y notation to row,col for clarity
        x_pixels = pixel_coords_2d[0, :]
        y_pixels = pixel_coords_2d[1, :]

        x = (x_pixels - cx) * depth / fx
        y = (y_pixels - cy) * depth / fy

        # Mujoco uses convention where z-axis points into the camera
        points_3d = np.vstack([x, y, -depth])

        return points_3d

    def _camera_to_world_frame(self, points_in_camera_frame: np.ndarray) -> np.ndarray:
        """
        Transform points from the camera frame to the world frame.

        Args:
            points_in_camera_frame: 3D points in camera frame as a (3, N) array.

        Returns:
            3D points in world frame as a (3, N) array.
        """
        t_reshaped = self.t[:, np.newaxis]  # t should be (3, 1)
        points_world = self.R @ points_in_camera_frame + t_reshaped
        return points_world
