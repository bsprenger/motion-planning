from typing import Literal

import cv2
import numpy as np

COLORS = Literal["red", "green", "blue"]


def create_color_mask(image: np.ndarray, color: COLORS) -> np.ndarray | None:
    """
    Args:
        image: The input RGB image as a NumPy array. TODO: check if RGB or BGR
        color: The color to segment.

    Returns:
        A binary mask where the specified color is white (255) and all other colors are black (0).
    """
    # TODO: double check RGB vs BGR
    if image.size == 0:
        return None

    # We convert to HSV for more intuitive color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # TODO: paremterize the color ranges
    if color == "red":
        # Red can span across 0-10 and 170-180 in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "green":
        lower_green = np.array([36, 50, 50])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
    elif color == "blue":
        lower_blue = np.array([100, 150, 0])  # Adjusted lower bound for blue
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    else:
        print(f"Invalid color: {color}. Valid options are 'red', 'green', 'blue'.")
        return None

    kernel = np.ones((5, 5), np.uint8)  # TODO: parameterize kernel size

    # Apply morphological operations to clean up the mask
    # Morphological opening removes noise (erodes then dilates)
    # Morphological closing fills small holes (dilates then erodes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
