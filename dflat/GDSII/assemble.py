import cv2
import numpy as np
from tqdm.auto import tqdm
import gdspy
import time


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
    assert len(cell_size) == 2
    assert len(block_size) == 2
    assert np.all(np.greater_equal(block_size, cell_size))
    assert len(params.shape) == 3
    assert len(mask.shape) == 2
    assert mask.shape == params.shape[0:2]
    assert params.shape[-1] == 1, "Shape dimension D should be equal to 1 (diameter)"

    # upsample the params to match the target blocks
    H, W, C = params.shape
    scale_factor = np.array(block_size) / np.array(cell_size)
    Hnew = np.rint(H * scale_factor[0]).astype(int)
    Wnew = np.rint(W * scale_factor[1]).astype(int)

    params_ = cv2.resize(
        params,
        (Wnew, Hnew),
        interpolation=cv2.INTER_LINEAR,
    )
    params_ = np.expand_dims(params_, -1)
    pshape = params_.shape
    mask = cv2.resize(
        np.expand_dims(mask, -1),
        (Wnew, Hnew),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = mask.astype(int).astype(bool)

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
                circle = gdspy.Round((xoffset, yoffset), radius, number_of_points=9)
                cell.add(circle)

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


def add_marker_tag(cell, ms, hx, hy):
    mw = ms * 5 / 9
    mh = ms / 2 + (21.10 / ms)  # Correction for the + sign center

    cx = -mw / 2
    cy = -mh + 2
    # Add alignment markers at
    cell.add(gdspy.Text("+", ms, (cx - ms / 2, cy - ms / 2)))  # lower left
    cell.add(gdspy.Text("+", ms, (cx + hx + ms / 2, cy - ms / 2)))  # lower right
    cell.add(gdspy.Text("+", ms, (cx + hx + ms / 2, cy + hx + ms / 2)))  # upper right
    cell.add(gdspy.Text("+", ms, (cx - ms / 2, cy + hx + ms / 2)))  # upper left

    return
