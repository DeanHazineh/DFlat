import numpy as np
from tqdm.auto import tqdm
import gdspy
import time

from .gds_utils import add_marker_tag, upsample_block


def assemble_nanocylinder_gds(
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
    """Generate a GDS for nanocylinder metasurfaces.

    Args:
        params (float): Nanocylinder radius across the lens of shape [H, W, 1].
        mask (int): Boolean mask of whether to write a shape or skip it of shape [H, W].
        cell_size (list): Cell sizes holding the nanocylinder of [dy, dx].
        block_size (list): Block sizes to repeat the nanocylinders of [dy', dx']. resize function is applied.
        savepath (str): Path to save the gds file (including .gds extension).
        gds_unit (flaot, optional): gdspy units. Defaults to 1e-6.
        gds_precision (float, optional): gdspy precision. Defaults to 1e-9.
        marker_size (float, optional): size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the shape. Defaults to 9.
    """
    assert (
        params.shape[-1] == 1
    ), "Shape dimension D encodes radius should be equal to 1."
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
    """Generate a GDS for Nano-ellipse metasurfaces.

    Args:
        params (float): Nanocylinder radius across the lens of shape [H, W, 1].
        mask (int): Boolean mask of whether to write a shape or skip it of shape [H, W].
        cell_size (list): Cell sizes holding the nanocylinder of [dy, dx].
        block_size (list): Block sizes to repeat the nanocylinders of [dy', dx']. resize function is applied.
        savepath (str): Path to save the gds file (including .gds extension).
        gds_unit (flaot, optional): gdspy units. Defaults to 1e-6.
        gds_precision (float, optional): gdspy precision. Defaults to 1e-9.
        marker_size (float, optional): size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the shape. Defaults to 9.
    """
    assert (
        params.shape[-1] == 2
    ), "Shape dimension D encodes radius (x,y) should be equal to 2."
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


def asseble_nanofin_gds(
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
    """Generate a GDS for Nanofin metasurfaces.

    Args:
        params (float): Nanocylinder radius across the lens of shape [H, W, 1].
        mask (int): Boolean mask of whether to write a shape or skip it of shape [H, W].
        cell_size (list): Cell sizes holding the nanocylinder of [dy, dx].
        block_size (list): Block sizes to repeat the nanocylinders of [dy', dx']. resize function is applied.
        savepath (str): Path to save the gds file (including .gds extension).
        gds_unit (flaot, optional): gdspy units. Defaults to 1e-6.
        gds_precision (float, optional): gdspy precision. Defaults to 1e-9.
        marker_size (float, optional): size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the shape. Defaults to 9.
    """
    assert (
        params.shape[-1] == 2
    ), "Shape dimension D encodes width should be equal to 1."
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
    assert len(cell_size) == 2
    assert len(block_size) == 2
    assert np.all(np.greater_equal(block_size, cell_size))
    assert len(params.shape) == 3
    assert len(mask.shape) == 2
    assert mask.shape == params.shape[0:2]

    # upsample the params to match the target blocks
    params_, mask = upsample_block(params, mask, cell_size, block_size)
    mask = mask.astype(int).astype(bool)
    pshape = params_.shape

    # Write to GDS
    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    cell = lib.new_cell("MAIN")
    print("Writing metasurface shapes to GDS File")
    start = time.time()
    for yi in tqdm(range(pshape[0])):
        for xi in range(pshape[1]):
            if mask[yi, xi]:
                xoffset = cell_size[1] * xi / gds_unit
                yoffset = cell_size[0] * yi / gds_unit
                radius = params_[yi, xi, 0] / gds_unit
                shape = cell_fun((xoffset, yoffset), radius, number_of_points=9)
                cell.add(shape)

    ### Add some lens markers
    hx = cell_size[1] * pshape[1] / gds_unit
    hy = cell_size[0] * pshape[0] / gds_unit
    ms = marker_size / gds_unit
    cell_annot = lib.new_cell("TEXT")
    add_marker_tag(cell_annot, ms, hx, hy)

    top_cell = lib.new_cell("TOP_CELL")
    # Reference cell1 and cell2 in the top-level cell
    ref_cell1 = gdspy.CellReference(cell)
    ref_cell2 = gdspy.CellReference(cell_annot)
    top_cell.add(ref_cell1)
    top_cell.add(ref_cell2)

    lib.write_gds(savepath)
    end = time.time()
    print("Completed writing and saving metasurface GDS File: Time: ", end - start)

    return
