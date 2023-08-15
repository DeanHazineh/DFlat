import torch


def torch_coordinate_grid(input_pixel_number, input_pixel_size_m, radial_symmetry, dtype):
    """Create the meshgrid coordinate grid from the grid parameters

    Args:
        input_pixel_number (dict): number of pixels in grid with keys "x", "y", "r"
        input_pixel_size_m (dict): pixel size (meters) with keys "x", "y"
        radial_symmetry (boolean): radial symmtery flag
        dtype (torch type): torch tensor dtype
    """

    if radial_symmetry:
        input_pixel_x, input_pixel_y = torch.meshgrid(
            torch.arange(input_pixel_number["r"], dtype=dtype),
            torch.arange(1, dtype=dtype),
            indexing="xy",
        )
    else:
        input_pixel_x, input_pixel_y = torch.meshgrid(
            torch.arange(input_pixel_number["x"], dtype=dtype),
            torch.arange(input_pixel_number["y"], dtype=dtype),
            indexing="xy",
        )
        input_pixel_x = input_pixel_x - (input_pixel_x.shape[1] - 1) / 2
        input_pixel_y = input_pixel_y - (input_pixel_y.shape[0] - 1) / 2
    input_pixel_x = input_pixel_x * input_pixel_size_m["x"]
    input_pixel_y = input_pixel_y * input_pixel_size_m["y"]

    return input_pixel_x, input_pixel_y
