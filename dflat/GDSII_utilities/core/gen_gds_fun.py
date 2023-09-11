import gdspy
import time
from tqdm.auto import tqdm


def add_marker_tag(cell, cell_size, meta_shape, gds_unit, add_markers):
    ### Add some lens markers (bottom and left)
    halfx = meta_shape[-1] // 2
    halfy = meta_shape[-2] // 2
    marker_span = 100e-6 / gds_unit
    cell_size_x = cell_size["x"]
    cell_size_y = cell_size["y"]

    x_loc = cell_size_x * halfx / gds_unit - marker_span / 3
    y_loc = cell_size_y * halfy / gds_unit - marker_span / 2

    if add_markers:
        htext = gdspy.Text("+", marker_span, (x_loc, -3 * marker_span))
        cell.add(htext)
        htext = gdspy.Text("+", marker_span, (x_loc, -6 * marker_span))
        cell.add(htext)

        htext = gdspy.Text("+", marker_span, (-3 * marker_span, y_loc))
        cell.add(htext)
        htext = gdspy.Text("+", marker_span, (-6 * marker_span, y_loc))
        cell.add(htext)

    ### Add some text above the lens
    text_height = 25e-6 / gds_unit
    y_loc = cell_size_y * meta_shape[-2] / gds_unit + 2 * text_height
    htext = gdspy.Text("DFlat", text_height, (x_loc, y_loc))
    cell.add(htext)

    return cell


def assemble_nanocylinder_gds(inputs, cell_size, savepath, gds_unit=1e-6, gds_precision=1e-9, add_markers=True):
    shape_array = inputs[0]
    rotation_array = inputs[1]
    boolean_mask = inputs[2].astype(bool)  # 1, Ny, Nx
    meta_shape = boolean_mask.shape
    cell_size_x = cell_size["x"]
    cell_size_y = cell_size["y"]

    print("Writing metasurface shapes to GDS File")
    start = time.time()

    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    cell = lib.new_cell("MAIN")

    for yi in tqdm(range(meta_shape[-2])):
        for xi in range(meta_shape[-1]):
            if boolean_mask[0, yi, xi]:
                xoffset = cell_size_x * xi / gds_unit
                yoffset = cell_size_y * yi / gds_unit

                radius = shape_array[0, yi, xi] / gds_unit

                # Create circle (projection of cylinder)
                circle = gdspy.Round((xoffset, yoffset), radius)
                cell.add(circle)

    ### Add some lens markers (bottom and left)
    cell = add_marker_tag(cell, cell_size, meta_shape, gds_unit, add_markers)
    lib.write_gds(savepath + "saved_lens_gdsII.gds")

    ###
    end = time.time()
    print("Completed writing and saving metasurface GDS File: Time: ", end - start)
    del cell
    return


def assemble_nanofins_gds(inputs, cell_size, savepath, gds_unit=1e-6, gds_precision=1e-9, add_markers=True):
    shape_array = inputs[0]
    rotation_array = inputs[1]
    boolean_mask = inputs[2].astype(bool)  # 1, Ny, Nx
    meta_shape = boolean_mask.shape
    cell_size_x = cell_size["x"]
    cell_size_y = cell_size["y"]

    print("Writing metasurface shapes to GDS File")
    start = time.time()

    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    cell = lib.new_cell("MAIN")

    for yi in tqdm(range(meta_shape[-2])):
        for xi in range(meta_shape[-1]):
            if boolean_mask[0, yi, xi]:
                xoffset = cell_size_x * xi / gds_unit
                yoffset = cell_size_y * yi / gds_unit

                Lx = shape_array[0, yi, xi] / gds_unit
                Ly = shape_array[1, yi, xi] / gds_unit

                rect = gdspy.Rectangle((-Lx / 2, -Ly / 2), (Lx / 2, Ly / 2))
                rect.rotate(rotation_array[0, yi, xi])
                rect.translate(xoffset, yoffset)
                cell.add(rect)

    ### Add some lens markers (bottom and left)
    cell = add_marker_tag(cell, cell_size, meta_shape, gds_unit, add_markers)
    lib.write_gds(savepath + "saved_lens_gdsII.gds")

    ###
    end = time.time()
    print("Completed writing and saving metasurface GDS File: Time: ", end - start)
    del cell

    return
