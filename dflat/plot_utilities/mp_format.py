import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize_text = 8.0
fontsize_title = 10.0
fontsize_ticks = 8.0
fontsize_cbar = 8.0
fontsize_legend = 8.0


def add_colorbar(
    fig,
    ax,
    im,
    title="",
    fs_cbar=fontsize_cbar,
    fs_ticks=fontsize_ticks,
):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    # cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(title, rotation=90, fontsize=fs_cbar)

    cbar.formatter.set_powerlimits((0, 0))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fs_ticks)

    return


def format_plot(
    fig,
    ax,
    xlabel="",
    ylabel="",
    title="",
    xvec=[],
    yvec=[],
    rmvxLabel=False,
    rmvyLabel=False,
    addcolorbar=False,
    cbartitle="",
    setxLim=[],
    setyLim=[],
    addLegend=False,
    fs_text=fontsize_text,
    fs_title=fontsize_title,
    fs_ticks=fontsize_ticks,
    fs_cbar=fontsize_cbar,
    fs_legend=fontsize_legend,
    setAspect="auto",
):
    imhandle = ax.images[0]
    ax.set_xlabel(xlabel, fontsize=fs_text)
    ax.set_ylabel(ylabel, fontsize=fs_text)
    ax.set_title(title, fontsize=fs_title)

    if len(xvec) != 0 and len(yvec) != 0:
        imhandle.set_extent([np.min(xvec), np.max(xvec), np.max(yvec), np.min(yvec)])

    if rmvxLabel:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    if rmvyLabel:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    if addcolorbar:
        add_colorbar(fig, ax, imhandle, cbartitle, fs_cbar, fs_ticks)
    else:
        divider2 = make_axes_locatable(ax)
        cax2 = divider2.append_axes("right", size="8%", pad=0.05)
        cax2.axis("off")

    if setxLim:
        ax.set_xlim(setxLim[0], setxLim[1])

    if setyLim:
        ax.set_ylim(setyLim[0], setyLim[1])

    if addLegend:
        legend = ax.legend(fontsize=fs_legend)

    # update fontsize for labels and ticks
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs_ticks)

    # Set aspect ratio
    ax.set_aspect(setAspect)


def axis_off(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Hide the ticks and tick labels
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    return
