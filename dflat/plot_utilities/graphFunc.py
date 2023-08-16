from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
from PIL import Image

# These lines are required to make fonts the right form for adobe illustrator
plt.rcParams["pdf.fonttype"] = 42.0
plt.rcParams["ps.fonttype"] = 42.0

fontsize_text = 10.0
fontsize_title = 12.0
fontsize_ticks = 10.0
fontsize_cbar = 10.0
fontsize_legend = 12.0


def addAxis(thisfig, n1, n2, maxnumaxis=""):
    axlist = []
    counterval = maxnumaxis if maxnumaxis else n1 * n2
    for i in range(counterval):
        axlist.append(thisfig.add_subplot(n1, n2, i + 1))

    return axlist


def addColorbar(
    thisfig,
    thisax,
    thisim,
    cbartitle="",
    fontsize_cbar=fontsize_cbar,
    fontsize_ticks=fontsize_ticks,
):
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    cbar = thisfig.colorbar(thisim, cax=cax, orientation="vertical")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cbartitle, rotation=90, fontsize=fontsize_cbar)

    cbar.formatter.set_powerlimits((0, 0))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_ticks)

    return


def formatPlots(
    thisfig,
    thisax,
    imhandle=None,
    xlabel="",
    ylabel="",
    title="",
    xgrid_vec=[],
    ygrid_vec=[],
    rmvxLabel=False,
    rmvyLabel=False,
    addcolorbar=False,
    cbartitle="",
    setxLim=[],
    setyLim=[],
    addLegend=False,
    fontsize_text=fontsize_text,
    fontsize_title=fontsize_title,
    fontsize_ticks=fontsize_ticks,
    fontsize_cbar=fontsize_cbar,
    fontsize_legend=fontsize_legend,
    setAspect="auto",
):
    # Set axis label and title
    thisax.set_xlabel(xlabel, fontsize=fontsize_text)
    thisax.set_ylabel(ylabel, fontsize=fontsize_text)
    thisax.set_title(title, fontsize=fontsize_title)

    if imhandle == None:
        imhandle = thisax.images[0]

    # Add x and y axis coordinates
    if len(xgrid_vec) != 0 and len(ygrid_vec) != 0:
        imhandle.set_extent([np.min(xgrid_vec), np.max(xgrid_vec), np.max(ygrid_vec), np.min(ygrid_vec)])

    # remove axis labels and ticks
    if rmvxLabel:
        thisax.set_xticklabels([""])
        thisax.set_xlabel("")

    if rmvyLabel:
        thisax.set_yticklabels([""])
        thisax.set_ylabel("")

    # add colorbar and preserve other subplot image sizes
    if addcolorbar:
        addColorbar(
            thisfig,
            thisax,
            imhandle,
            cbartitle,
            fontsize_cbar=fontsize_cbar,
            fontsize_ticks=fontsize_ticks,
        )
    else:
        # This is useful to change the axis size such that the axis with and without colorbar is the same shape
        divider2 = make_axes_locatable(thisax)
        cax2 = divider2.append_axes("right", size="8%", pad=0.05)
        cax2.axis("off")

    if setxLim:
        thisax.set_xlim(setxLim[0], setxLim[1])

    if setyLim:
        thisax.set_ylim(setyLim[0], setyLim[1])

    if addLegend:
        legend = thisax.legend(fontsize=fontsize_legend)

    # update fontsize for labels and ticks
    for item in thisax.get_xticklabels() + thisax.get_yticklabels():
        item.set_fontsize(fontsize_ticks)

    # Set aspect ratio
    thisax.set_aspect(setAspect)

    return


def gif_from_saved_images(filepath, filetag, savename, fps, deleteFrames=True, verbose=False):
    print("Call GIF generator")
    images = []

    png_files = [f for f in os.listdir(filepath) if f.startswith(filetag) and f.endswith(".png")]
    png_files = sorted(png_files, key=lambda f: os.path.getmtime(os.path.join(filepath, f)))
    for file in png_files:
        file_path = os.path.join(filepath, file)
        images.append(Image.open(file_path))

        if verbose:
            print("Write image file as frame: " + file)
        if deleteFrames:
            os.remove(file_path)

    duration = int(1000 / fps)
    images[0].save(
        filepath + savename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    return
