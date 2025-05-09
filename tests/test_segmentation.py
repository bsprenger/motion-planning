import numpy as np
import pytest

from motion_planning.segmentation import create_color_mask


@pytest.fixture
def sample_image():  # TODO double check if it should be RGB or BGR
    # Create a simple 100x100 BGR image with three colored squares
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Red square
    image[10:40, 10:40] = [0, 0, 255]
    # Green square
    image[10:40, 50:80] = [0, 255, 0]
    # Blue square
    image[50:80, 10:40] = [255, 0, 0]
    return image


def test_segment_red_cube(sample_image):
    mask = create_color_mask(sample_image, "red")
    assert mask is not None
    assert np.sum(mask[10:40, 10:40] == 255) > (30 * 30 * 0.8)
    assert np.sum(mask[10:40, 50:80] == 255) < (30 * 30 * 0.2)
    assert np.sum(mask[50:80, 10:40] == 255) < (30 * 30 * 0.2)


def test_segment_green_cube(sample_image):
    mask = create_color_mask(sample_image, "green")
    assert mask is not None
    assert np.sum(mask[10:40, 50:80] == 255) > (30 * 30 * 0.8)
    assert np.sum(mask[10:40, 10:40] == 255) < (30 * 30 * 0.2)
    assert np.sum(mask[50:80, 10:40] == 255) < (30 * 30 * 0.2)


def test_segment_blue_cube(sample_image):
    mask = create_color_mask(sample_image, "blue")
    assert mask is not None
    assert np.sum(mask[50:80, 10:40] == 255) > (30 * 30 * 0.8)
    assert np.sum(mask[10:40, 10:40] == 255) < (30 * 30 * 0.2)
    assert np.sum(mask[10:40, 50:80] == 255) < (30 * 30 * 0.2)


def test_invalid_color(sample_image):
    mask = create_color_mask(sample_image, "yellow")
    assert mask is None


def test_empty_image():
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert create_color_mask(empty_img, "red") is None


def test_image_with_no_target_color(sample_image):
    image_only_blue = np.zeros((100, 100, 3), dtype=np.uint8)
    image_only_blue[50:80, 10:40] = [255, 0, 0]  # Blue square only

    mask_red = create_color_mask(image_only_blue, "red")
    assert mask_red is not None
    assert np.sum(mask_red == 255) == 0
