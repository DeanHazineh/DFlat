import numpy as np
from tqdm.auto import tqdm
import gdspy
import time
import uuid

from .gds_utils import add_marker_tag, upsample_block


def assemble_cylinder_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nanocylinder metasurfaces.

    Args:
        params (numpy.ndarray): Nanocylinder radii across the lens, shape [H, W, 1].
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nanocylinder [dy, dx].
        block_size (list): Block sizes to repeat the nanocylinders [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the circular shape. Defaults to 9.

    Raises:
        ValueError: If params.shape[-1] != 1.
    """
    if params.shape[-1] != 1:
        raise ValueError("Shape dimension D encoding radius should be equal to 1.")

    assemble_standard_shapes(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_ellipse_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nano-ellipse metasurfaces.

    Args:
        params (numpy.ndarray): Ellipse radii across the lens, shape [H, W, 2] where [:,:,0] is x-radius and [:,:,1] is y-radius.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nano-ellipse [dy, dx].
        block_size (list): Block sizes to repeat the nano-ellipses [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the elliptical shape. Defaults to 9.

    Raises:
        ValueError: If params.shape[-1] != 2.
    """
    if params.shape[-1] != 2:
        raise ValueError("Shape dimension D encoding radii (x,y) should be equal to 2.")

    assemble_standard_shapes(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_fin_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nanofin metasurfaces.

    Args:
        params (numpy.ndarray): Nanofin dimensions across the lens, shape [H, W, 2] where [:,:,0] is width and [:,:,1] is length.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nanofin [dy, dx].
        block_size (list): Block sizes to repeat the nanofins [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.

    Raises:
        ValueError: If params.shape[-1] != 2.
    """
    if params.shape[-1] != 2:
        raise ValueError(
            "Shape dimension D encoding width and length should be equal to 2."
        )

    assemble_standard_shapes(
        gdspy.Rectangle,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_standard_shapes(
    cell_fun,
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """
    Assemble standard shapes for GDS files based on given parameters.

    This function creates a GDS file containing a metasurface pattern of standard shapes
    (e.g., circles, ellipses, rectangles) based on the provided parameters and mask.

    Args:
        cell_fun (callable): GDSPY function to create the shape (e.g., gdspy.Round, gdspy.Rectangle).
        params (numpy.ndarray): Shape parameters across the lens, shape [H, W, D] where D depends on the shape type.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the shape [dy, dx].
        block_size (list): Block sizes to repeat the shapes [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent curved shapes. Defaults to 9.

    Raises:
        ValueError: If input dimensions are incorrect or inconsistent.

    Returns:
        None
    """
    # Input validation
    if len(cell_size) != 2 or len(block_size) != 2:
        raise ValueError("cell_size and block_size must be lists of length 2.")
    if not np.all(np.greater_equal(block_size, cell_size)):
        raise ValueError("block_size must be greater than or equal to cell_size.")
    if len(params.shape) != 3 or len(mask.shape) != 2:
        raise ValueError("params must be 3D and mask must be 2D.")
    if mask.shape != params.shape[:2]:
        raise ValueError("mask shape must match the first two dimensions of params.")

    # Upsample the params to match the target blocks
    params_, mask = upsample_block(params, mask, cell_size, block_size)
    mask = mask.astype(bool)
    pshape = params_.shape

    # Write to GDS
    unique_id = str(uuid.uuid4())[:8]
    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    cell = lib.new_cell(f"MAIN_{unique_id}")
    print("Writing metasurface shapes to GDS File")
    start = time.time()

    for yi, xi in np.ndindex(pshape[:2]):
        if mask[yi, xi]:
            xoffset = cell_size[1] * xi / gds_unit
            yoffset = cell_size[0] * yi / gds_unit
            shape_params = params_[yi, xi] / gds_unit
            shape_params = shape_params.flatten()

            ## In new version of GDSPY, we can no longer specify rectangle widths (?)
            ## Now it corresponds to edge coordintes which is unfortunate
            if cell_fun == gdspy.Round:
                if len(shape_params) == 1:
                    shape_params = [shape_params[0], shape_params[0]]
                shape = cell_fun((xoffset, yoffset), shape_params)
            elif cell_fun == gdspy.Rectangle:
                shape_params += [xoffset, yoffset]
                shape = cell_fun((xoffset, yoffset), shape_params)
            else:
                raise ValueError
            cell.add(shape)

    # Add lens markers
    hx = cell_size[1] * pshape[1] / gds_unit
    hy = cell_size[0] * pshape[0] / gds_unit
    ms = marker_size / gds_unit
    cell_annot = lib.new_cell(f"TEXT_{unique_id}")
    add_marker_tag(cell_annot, ms, hx, hy)

    # Create top-level cell and add references
    top_cell = lib.new_cell(f"TOP_CELL_{unique_id}")
    top_cell.add(gdspy.CellReference(cell))
    top_cell.add(gdspy.CellReference(cell_annot))

    # Write GDS file
    lib.write_gds(savepath)
    end = time.time()
    print(
        f"Completed writing and saving metasurface GDS File. Time: {end - start:.2f} seconds"
    )

    return


if __name__ == "__main__":
    assemble_fin_gds(
        np.random.rand(10, 10, 2) * 250e-9,
        np.random.choice([True, False], size=(10, 10)),
        [500e-9, 500e-9],
        [1e-6, 1e-6],
        "/home/deanhazineh/Research/DFlat/dflat/GDSII/out.gds",
        gds_unit=1e-6,
        gds_precision=1e-9,
        marker_size=250e-6,
        number_of_points=9,
    )
