import numpy as np
from scipy.ndimage import zoom


def generate_idmask_sieve(num_sets, lens_size):
    # Determine the dimensions of the unit cell
    root_num_sets = np.sqrt(num_sets)
    if np.round(root_num_sets) == root_num_sets:
        unit_cell_len_x = int(root_num_sets)
        unit_cell_len_y = int(root_num_sets)
    elif np.round(root_num_sets) > root_num_sets:
        unit_cell_len_x = int(np.ceil(root_num_sets))
        unit_cell_len_y = unit_cell_len_x
    else:
        unit_cell_len_x = int(np.ceil(root_num_sets))
        unit_cell_len_y = int(np.floor(root_num_sets))

    # Calculate the number of elements in a cell and create the base unit
    num_in_cell = unit_cell_len_x * unit_cell_len_y
    base_unit = np.arange(1, num_sets + 1)
    base_vector = base_unit
    while len(base_vector) < num_in_cell:
        base_vector = np.concatenate((base_vector, base_unit))
    base_vector = base_vector.flatten()
    base_vector = np.reshape(
        base_vector[:num_in_cell], (unit_cell_len_x, unit_cell_len_y)
    )

    # Tile the repeating unit to cover the desired size
    num_reps_x = int(lens_size[1] / unit_cell_len_x + 1)
    num_reps_y = int(lens_size[0] / unit_cell_len_y + 1)
    tile_matrix = np.tile(base_vector, (num_reps_x, num_reps_y))

    # Trim the tiled matrix to fit the exact lens_size dimensions
    tile_matrix = tile_matrix[: lens_size[0], : lens_size[1]]

    return tile_matrix


def multiplexing_mask_sieve(num_sets, lens_size):
    """
    Generate a stack of checkerboard-like, binary masks, useful for designing spatially multiplexed metasurfaces.

    Note: This function produces an ideal interleave behavior (equal energy across masks) only for values of num_sets
    that are perfect squares, e.g., 4, 9. In other cases, it is better to use random, orthogonal binary masks.

    Args:
        num_sets (int): Number of orthogonal masks to generate.
        lens_size (tuple): Dimensions of the lens.

    Raises:
        AssertionError: If num_sets is not an integer or less than 1, or if num_sets is not a perfect square when ideal behavior is expected.

    Returns:
        np.ndarray: A set of sieve binary masks, of shape (num_sets, lens_size[0], lens_size[1]).
    """

    assert isinstance(
        num_sets, int
    ), "multiplexing_mask_sieve: num_sets must be an integer."
    assert num_sets >= 1, "multiplexing_mask_sieve: num_sets should be greater than 1"
    assert (
        int(np.floor(np.sqrt(num_sets)) ** 2) == num_sets
    ), "Warning: Ideal sampling behavior only occurs for num_sets with integer root."
    assert len(lens_size) == 2, "Len_size should be a tuple like [H, W]."

    id_matrix = generate_idmask_sieve(num_sets, lens_size)
    mask_stack = []
    for i in range(num_sets):
        this_mask = np.copy(id_matrix)
        this_mask[np.where(this_mask != i + 1)] = 0
        this_mask[np.where(this_mask == i + 1)] = 1
        mask_stack.append(this_mask)

    return np.stack(mask_stack)


def generate_idmask_CodedAperture(num_sets, block_size_m, lens_dx_m, lens_size):
    """
    Generate a random ID mask with downsampling and upsampling to match the lens size.

    Args:
        num_sets (int): Number of sets or unique IDs to generate within the mask.
        block_size_m (tuple): The block size in meters, specified as (height, width).
        lens_dx_m (tuple): Pixel pitch of the lens in meters, specified as (height, width).
        lens_size (tuple): Size of the lens, specified as (height, width).

    Returns:
        np.ndarray: A downsampled and then upsampled ID matrix matching the lens size.
    """
    # Calculate the ratio of lens pixel pitch to block size for downsampling
    ratio_x = block_size_m[1] / lens_dx_m[1]
    ratio_y = block_size_m[0] / lens_dx_m[0]
    downsampled_shape = (int(lens_size[1] / ratio_y), int(lens_size[0] / ratio_x))

    # Generate a random ID matrix with the downsampled shape
    id_matrix = np.random.randint(1, num_sets + 1, size=downsampled_shape)
    id_matrix = zoom(id_matrix, (ratio_y, ratio_x), order=0)
    id_matrix = id_matrix[: lens_size[0], : lens_size[1]]

    return id_matrix


def multiplexing_mask_orthrand(num_sets, block_dx, lens_dx, lens_size):
    """
    Generates a set of orthogonal, random coded binary masks on top of the lens grid.

    Args:
        num_sets (int): Number of orthogonal masks to generate.
        block_dx (tuple): Smallest binary segment of the coded aperture, defined as (dx_y, dx_x).
        lens_dx (tuple): Pixel pitch of the lens, defined as (height, width).
        lens_size (tuple): Size of the lens, specified as (height, width).

    Returns:
        np.ndarray: Binary mask stack, of shape (num_sets, lens_size[0], lens_size[1]).
    """
    # Validate inputs
    assert isinstance(num_sets, int), "num_sets must be an integer"
    assert num_sets >= 1, "num_sets should be greater than 1"
    assert all(
        np.array(lens_dx) <= np.array(block_dx)
    ), "block_size_m of the mask cannot have smaller sampling than the lens grid"
    assert (np.mod(block_dx[0] / lens_dx[0], 1) == 0) and (
        np.mod(block_dx[1] / lens_dx[1], 1) == 0
    ), "Block size of mask should be an integer multiple of pixel pitch"

    # Generate a random ID matrix
    id_matrix = generate_idmask_CodedAperture(num_sets, block_dx, lens_dx, lens_size)
    mask_stack = []

    # Create binary masks from the ID matrix
    for i in range(num_sets):
        this_mask = np.copy(id_matrix)
        this_mask[np.where(this_mask != i + 1)] = 0
        this_mask[np.where(this_mask == i + 1)] = 1
        mask_stack.append(this_mask)

    return np.stack(mask_stack)
