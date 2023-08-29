from __future__ import annotations

from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(im, ax, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(im, cax=cax, orientation='vertical')
