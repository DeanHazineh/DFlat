from pathlib import Path
import numpy as np
from .core.load_image_fun import load_png_as_grayscale, scale_and_zero_pad


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "core/grayscale_test_images/"
    return resource_path.joinpath(file_name)


def get_grayscale_image(img_name, img_dim=None, resize_method="crop"):
    """Return image as grayscale image, resized to img_dim while preserving the original aspect ratio.

    Args:
        `img_name`: Filename in path datasets/grayscale_images/x.png
        `img_dim` (dict, optional): Dictionary with key "x" and "y", denoting number of pixels along each direction
        `resize_method` (str, optional): Either preserve aspect ratio by "crop" or by "pad"

    Returns:
        float: image
    """
    return load_png_as_grayscale(get_path_to_data(img_name), img_dim, resize_method)


def load_grayscale_fromPath(img_path, img_dim=None, resize_method="crop", shrink_img_scale=1.0, invert=False):
    """Return image as grayscale image, resized to img_dim while preserving the original aspect ratio.

    Args:
        `img_path`: filepath
        `img_dim` (dict, optional): Dictionary with key "x" and "y", denoting number of pixels along each direction
        `resize_method` (str, optional): Either preserve aspect ratio by "crop" or by "pad"
        'shrink_img_scale' (float, optional): Resize the png by a scale factor and zero pad so it is smaller
    Returns:
        float: image
    """
    img = load_png_as_grayscale(img_path, img_dim, resize_method)
    if invert:
        img = np.abs(img - np.max(img))

    if not np.isclose(shrink_img_scale, 1.0, 0.01):
        img = scale_and_zero_pad(img, shrink_img_scale)

    return img
