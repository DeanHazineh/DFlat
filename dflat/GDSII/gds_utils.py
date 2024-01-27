import numpy as np
import gdspy
import cv2


def upsample_block(params, mask, cell_size, block_size):
    # upsample the params to match the target blocks
    H, W, C = params.shape
    scale_factor = np.array(block_size) / np.array(cell_size)
    Hnew = np.rint(H * scale_factor[0]).astype(int)
    Wnew = np.rint(W * scale_factor[1]).astype(int)

    params = cv2.resize(
        params,
        (Wnew, Hnew),
        interpolation=cv2.INTER_LINEAR,
    )
    params = np.expand_dims(params, -1)

    mask = cv2.resize(
        np.expand_dims(mask, -1),
        (Wnew, Hnew),
        interpolation=cv2.INTER_LINEAR,
    )
    return params, mask


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
