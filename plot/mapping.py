__all__ = [
        #'load_countries',
        #'load_provinces',
        ]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_countries(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        edgecolor='k',
        facecolor='none',
        ):
    import cartopy.feature as feature
    return feature.NaturalEarthFeature(category=category, name=name, scale=scale, facecolor=facecolor, edgecolor=edgecolor)

def load_provinces(
        category='cultural',
        name='admin_1_state_provinces_lines',
        scale='10m',
        edgecolor='gray',
        facecolor='none',
        ):
    import cartopy.feature as feature
    return feature.NaturalEarthFeature(category=category, name=name, scale=scale, facecolor=facecolor, edgecolor=edgecolor)

def add_gridlines(ax,labels=True,rotation=25):
    """
    :param: ax, matplotlib.pyplot.axes object where you want the gridlines
            must have a transformation with cartopy, I guess
    """
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl = ax.gridlines(draw_labels=labels, color='k', alpha=0.2, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'rotation': rotation}
    gl.ylabel_style = {'rotation': rotation}
    return None

def map_cbar(fig, ax, width=0.025):
    """
    :param: fig, matplotlib.pyplot.figure object
    :param: ax, matplotlib.pyplot.axes object
    """
    fig.subplots_adjust(right=0.8)
    cbar_xmin = ax.get_position().xmax+0.05
    cbar_ymin = ax.get_position().ymin
    cbar_width = 0.025
    cbar_hight = ax.get_position().ymax-ax.get_position().ymin
    cbar_ax = fig.add_axes([cbar_xmin, cbar_ymin, cbar_width, cbar_hight])
    return cbar_ax
