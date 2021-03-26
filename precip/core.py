
__all__ = ['calc_precip_acum',
           'plot_precip',
           'ppmean_box',
           'd_format',
           'calc_running_acum',
          ]

##############################################################################
def calc_precip_acum(case, acum):
    """
    Calculates the FORWARD accumulated precipitation of both dataset or simulation

    - case: Docpy.Dato or Docpy.WRF object
    - acum: int, hours of accumulation. 

    - return: numpy array of dims (t,lat,lon)

    ej: if acum = 3, precip_acum at t=00Z  represents the precipitation accumulated from time 00Z to time 03Z. 
    """
    import numpy as np
    lat, lon = case.get_latlon()
    label_tiempos, delta_t_hs = case.timedata()

    len_tiempo = len(label_tiempos)
    inter = int(acum/delta_t_hs)

    if case.name == 'cmorph':
        rain = case.precip_selection()/2 #is in mm/30 mins 
    elif case.name == 'mswep':
        rain = case.precip_selection()
    elif case.name == 'wrf' :
        rain = np.array(case.get_variables('rainnc')[:,0,:])+np.array(case.get_variables('rainc')[:,0,:])+np.array(case.get_variables('rainsh')[:,0,:])
    elif case.name == 'estaciones':
        rain = case.precip_selection()
    elif case.name == 'imerg':
        rain = case.precip_selection()/2
    precip_acum = np.ma.empty([len(label_tiempos[::inter]),rain.shape[1],rain.shape[2]])

    for r in range(precip_acum.shape[0]-1):
        if case.__class__.__name__ == 'Dato':
            precip_acum[r,:,:] = np.ma.sum(rain[r*inter:(r+1)*inter,:,:], axis=0)
        elif case.__class__.__name__ == 'WRF':
            precip_acum[r,:,:] = rain[(r+1)*inter,:]-rain[r*inter,:]

    return precip_acum

###############################################################################
def ppmean_box(case, acum, box,offset=0):
    """
    Calculates the mean boxed forward accumulated precipitation

    - case:     DatosPy.Dato or WRFpyCIMA.CIMA object
    - acum:     int, hours of accumulation
    - box :     list, box of [lonmin, lonmax, latmin, latmax] dimensions
    - offset:   float, offset from which the precipitation for the mean is taken into account. Default is 0. 
    - return: numpy.array, series of precipitation
    """
    from . import calc_precip_acum
    import numpy as np
    precip_acum = calc_precip_acum(case, acum)
    lat, lon = case.get_latlon()                    
    lonmin, lonmax, latmin, latmax = box

    x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
    x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1              
    y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])            
    y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1
    
    if y_idx_ini>y_idx_fin:
        y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini
    precip_box = precip_acum[:,y_idx_ini:y_idx_fin,x_idx_ini:x_idx_fin]     
    media = np.zeros(precip_box.shape[0])
    for t in range(precip_box.shape[0]):
        media[t] = np.mean(precip_box[t,:,:][precip_box[t,:,:]>offset])
    return media

###############################################################################
def plot_precip(case, acum, ruta_figs, name_fig):
    """ 
    Accumulated precipitation plot.

    - case: DatosPy.Dato or WRFpyCIMA.CIMA object
    - acum: int, hours of accumulation
    - ruta_figs: str, path to figures directory
    - name_fig: str, name of the figure
    """
    print('###','Packaching','###')
    import DocPy
    from . import calc_precip_acum
    # paquetes para graficar
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    # demases
    import os
    print('###','Getting dimensional info','###')
    lat, lon = case.get_latlon()
    label_tiempos, delta_t_hs = case.timedata()
    ## Dominio común figuras ##
    print('###','Agarro la cajita','###')
    if label_tiempos[0][:4] == '2015':
        lonmin, lonmax, latmin, latmax = DocPy.utils['box15']
    else:
        lonmin, lonmax, latmin, latmax = DocPy.utils['box']
    print('###','Settings','###')
    paleta = plt.get_cmap('gist_ncar')
    paleta.set_under('white')
    dlon = 2
    dlat = 1.5
    dict_levels = {24: np.array([1, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 250]),
                    6: np.array([1, 2, 5, 10, 15, 20, 30, 50, 70, 90, 120]),
                    3: np.array([1, 2, 5, 10, 15, 20, 30, 50, 70, 90, 120])}
    
    box_extent = [lonmin, lonmax, latmin, latmax]
    levels = dict_levels[acum]
    str_levels=["" for x in range(levels.shape[0])]
    for i in range(levels.shape[0]):
        str_levels[i]='%d' % (levels[i])
    provincias = shpreader.Reader('/home/martin.feijoo/provincias/provincias.shp')

    print('###','Grafico','###')
    precip = calc_precip_acum(case, acum)
    for r in range(precip.shape[0]):
        print(label_tiempos[r*int(acum/delta_t_hs)])

        fig, ax = plt.subplots(figsize=(10,6),subplot_kw=dict(projection=ccrs.PlateCarree()))
        C=ax.contourf(lon,lat,precip[r,:,:],
                      transform=ccrs.PlateCarree(),
                      cmap=paleta,extend='both',levels=levels)
        cbar=plt.colorbar(C,ticks=levels)
        cbar.set_label('mm',fontsize=18)
        cbar.ax.yaxis.set_ticklabels(str_levels,fontsize=18)
        ax.set_extent(box_extent)
        for rec in provincias.records():
            ax.add_geometries( [rec.geometry], ccrs.PlateCarree(), edgecolor="k", facecolor='none', linewidth=0.5)
        gl = ax.gridlines(draw_labels=True,color='black',alpha=0.2,linestyle='--')
        gl.xlabels_top =False
        gl.ylabels_right = False
        gl.xlocator = mtick.FixedLocator(np.arange(int(np.round(lon.min())),int(np.round(lon.max()))+dlon,dlon))
        gl.ylocator = mtick.FixedLocator(np.arange(int(np.round(lat.min())),int(np.round(lat.max()))+dlat,dlat))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15}
        gl.ylabel_style = {'size': 15}
        ax.coastlines(resolution='50m')
        titulo='PP (mm) Acum '+str(acum)+'hs '+label_tiempos[int(acum/delta_t_hs)*r][8::]
        if case.__class__.__name__ == 'Dato':
            titulo = case.name.append(titulo)
        plt.title(titulo,fontsize=18)
        ##################################
        for fmt in ['.jpg']: #,'.pdf']:
            fig.savefig(os.path.join(ruta_figs,'_'.join([name_fig,'acum {:02d}'.format(acum),case.dominio,label_tiempos[r*int(acum/delta_t_hs)]])+fmt), dpi=150, bbox_inches='tight')
        plt.close()

d_format = {
        'levels':{24: [15, 30, 50, 80, 100, 125, 150, 175, 200, 250, 300],
                   6: [ 1,  5, 10, 15,  20,  30,  50,  70,  90, 120, 150],
                   3: [ 1,  5, 10, 15,  20,  30,  45,  60,  80, 100]}
        }

#################################################################################
def calc_running_acum(case, acum, run):
    """
    Calculates the FORWARD running accumulated precipitation of both dataset or simulation

    - case: Docpy.Dato or Docpy.WRF object
    - acum: int, hours of accumulation. 
    - run: int, hours of running accumulation. Must be beq than acum
    - return: numpy array of dims (t,lat,lon)

    ej: if acum = 3, precip_acum at t=00Z  represents the precipitation accumulated from time 00Z to time 03Z. 
    """
    from . import calc_precip_acum
    import numpy as np
    pp = calc_precip_acum(case, acum)
    
    times_of_acum = pp.shape[0]

    inter = run//acum

    rain = np.zeros((pp.shape[0]+1-inter,)+pp.shape[1:])
    for t in range(rain.shape[0]):
        rain[t,:,:] = np.sum( pp[t:t+inter,:,:], axis=0)

    return rain
