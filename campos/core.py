
__all__ = ['plot_z_lvl',
           'calc_z_lvl',
           'calc_cfhi',
           'calc_Vq_integrated',
           'vert_profile',
           ]

def calc_z_lvl(case, lvl):
    """
    Calculates the geopotential hight at the desired level

    - case: Docpy.WRF or Docpy.ERA5 class object
    - lvl: int, pressure level

    - return: numpy.array with dimensions (time, lat, lon)
    """
    g = 9.81
    lev = case.get_levels()
    if case.name == 'wrf':
        geopt = 'geopt'
    elif case.name == 'era5':
        geopt = 'z'
    elif case.name == 'erai':
        geopt = 'Z'#case.dvars_era['Z']
    z = case.get_variables(geopt)[:, (list(lev).index(lvl)),:,:]/g

    return z

def calc_cfhi(case, delta_t=3, smooth=True):
    """
    Calculates the Vertically Integrated Moisture Flux Convergence

    - case: Docpy.WRF class object
    - delta_t: in, time interval in hours in which you want the VIMFC to be calculated
    - smooth: float or True/False, smooth parameter. Generally between 0.5 and 1.6. default is True

    - return: numpy.array (time, lat, lon)
    """
    import numpy as np
    g = 9.80665
    a = 6371010
    lvls = case.get_levels()
    times, delta_t_hs = case.timedata()

    dvars = {
            'wrf':{
                'u':'umet',
                'v':'vmet',
                'qvapor':'qvapor'},
            'era5':{
                'u':'u',
                'v':'v',
                'q':'q'},
            }
    u_name = dvars[case.name]['u']
    v_name = dvars[case.name]['v']
    ua = case.get_variables('umet')[::int(delta_t/delta_t_hs),:,:,:]
    va = case.get_variables('vmet')[::int(delta_t/delta_t_hs),:,:,:]
    if case.name == 'wrf':
        qvapor = case.get_variables('qvapor')[::int(delta_t/delta_t_hs),:,:,:]
        hus = qvapor/(1+qvapor)
        del qvapor
    elif case.name == 'era5':
        hus = case.get_variables('q')
    qu = ua*hus
    qv = va*hus
    del hus
    # calculo la integral
    dps = np.diff(lvls[::-1])
    capax = np.zeros_like(qu)
    capay = np.zeros_like(qv)
    for l in range(len(dps)):
        capax[:,l,:,:] = (qu[:,l,:,:]+qu[:,l+1,:,:])*0.5*dps[l]*100/g
        capay[:,l,:,:] = (qv[:,l,:,:]+qv[:,l+1,:,:])*0.5*dps[l]*100/g

    vimfx = np.sum(capax[:,:-1,:,:], axis=1)
    vimfy = np.sum(capay[:,:-1,:,:], axis=1)
    del capax, capay

    lat, lon = case.get_latlon()
    chifx = chify = np.empty(vimfx.shape)
    for t in range(vimfx.shape[0]):
        for j in range(len(lat)):
            chifx[t,j,:] = np.gradient(vimfx[t,j,:])/np.gradient(np.deg2rad(lon))
        for i in range(len(lon)):
            chify[t,:,i] = np.gradient(vimfy[t,:,i])/np.gradient(np.deg2rad(lat))
    del vimfx, vimfy
    chif = np.empty(chifx.shape)
    for j in range(len(lat)):
        chif[:,j,:] = (chifx[:,j,:]+chify[:,j,:])/(a*np.cos(np.deg2rad(lat[j])))
    
    if smooth != False:
        import scipy.ndimage as sci
        if smooth == True:
            a = -8.75
            b = 1.55
            r = (np.diff(lon)[0]+np.diff(lat)[0])/2
            f = lambda x: a*x+b
            chif = sci.filters.gaussian_filter(chif, f(r))
        chif = sci.filters.gaussian_filter(chif, smooth)
    return chif

def calc_Vq_integrated(case, delta_t):
    """
    Calculates the Vertically Integrated Moisture Flux 
    - case: Docpy.WRF,  Docpy.ERA5  or Docpy.ERAI class object
    - delta_t: in, time interval in hours in which you want the VIMFC to be calculated
    
    - return: vimfx, vimfy. numpy.array (time, lat, lon)
    """
    import numpy as np
    g = 9.80665 #m/s-1
    a = 6371010 # m
    lvls = case.get_levels()
    times, delta_t_hs = case.timedata()
    
    dvars = {
            'wrf':{
                'u':'umet',
                'v':'vmet',
                'q':'qvapor'},
            'era5':{
                'u':'u',
                'v':'v',
                'q':'q'},
            'erai':{
                'u':'U',
                'v':'V',
                'q':'HUS'},
            }
    u_name = dvars[case.name]['u']
    v_name = dvars[case.name]['v']
    q_name = dvars[case.name]['q']
    ua = case.get_variables(u_name)[::int(delta_t/delta_t_hs),:,:,:] #m/s
    va = case.get_variables(v_name)[::int(delta_t/delta_t_hs),:,:,:]
    if case.name=='wrf':
        qvapor = case.get_variables(q_name)[::int(delta_t/delta_t_hs),:,:,:] #kg/kg
        hus = qvapor/(1+qvapor) #kg/kg
        del qvapor
    elif (case.name=='era5') or (case.name=='erai'):
        hus = case.get_variables(q_name)[::int(delta_t/delta_t_hs),:,:,:]

    qu = ua*hus #m kg / s kg
    qv = va*hus
    del hus
    # calculo la integral
    dps = np.diff(lvls)#[::-1]) #hpa
    capax = np.zeros_like(qu)
    capay = np.zeros_like(qv)
    
    if case.name == 'erai':
        factor = 100
    else:
        factor = 100
    for l in range(len(dps)):
        capax[:,l,:,:] = (qu[:,l,:,:]+qu[:,l+1,:,:])*0.5*np.abs(dps[l])*factor/g #kg/m s 
        capay[:,l,:,:] = (qv[:,l,:,:]+qv[:,l+1,:,:])*0.5*np.abs(dps[l])*factor/g
        
    vimfx = np.sum(capax[:,:-1,:,:], axis=1)
    vimfy = np.sum(capay[:,:-1,:,:], axis=1)
    del capax, capay
    
    return vimfx, vimfy

def plot_z_lvl(case, delta_t, lvl, ruta_figs, name_fig):
    """
    Plots the geopotential hight in any pressure level you want

    - case: WRFpyCIMA.CIMA class object
    - delta_t: int, time interval in hours in which you want the plots to be made
    - lvl: int, pressure level. 
    - ruta_figs: string, path to save files
    """
    import os
    import cartopy.crs as ccrs        
    import cartopy.io.shapereader as shpreader          
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    lat, lon = case.get_latlon()
    lev = case.get_levels()

    g = 9.81

    z = case.get_variables('geopt')[:, (list(lev).index(lvl)),:,:]/g
    label_tiempos,delta_t_hs = case.timedata()
    # configureta
    paleta=plt.get_cmap('YlGnBu')
    paleta.set_under('white')

    dlon = round(round(lon.max()-lon.min())/6)
    dlat = round(round(lat.max()-lat.min())/6)

    levels = [int(i) for i in np.linspace(1250,1550,11)]

    str_levels=["" for x in range(len(levels))]
    for i in range(len(levels)):
        str_levels[i]='%d' % (levels[i])
    provincias = shpreader.Reader('/home/martin.feijoo/provincias/provincias.shp')

    for t in range(0,z.shape[0], int(delta_t/delta_t_hs)):
        print(label_tiempos[t])
        fig = plt.figure(figsize=(10,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        cs = ax.contour(lon, lat, z[t,:,:], transform=ccrs.PlateCarree(), levels=levels, colors='k', linewidths=0.8)
        c = ax.contourf(lon, lat, z[t,:,:], transform=ccrs.PlateCarree(), cmap=paleta, extend='both', levels=levels)

        cbar = plt.colorbar(c, ticks=levels, format='%d')
        cbar.set_label('m', fontsize=14)
        cbar.ax.yaxis.set_ticklabels(str_levels, fontsize=14)
        
        #ax.set_extent([min(lon), max(lon), min(lat), max(lat)])
        for rec in provincias.records():
            ax.add_geometries( [rec.geometry], ccrs.PlateCarree(), edgecolor="k", facecolor='none', linewidth=0.5)
        gl = ax.gridlines(draw_labels=True,color='black',alpha=0.2,linestyle='--')
        gl.xlabels_top =False
        gl.ylabels_right = False
        gl.xlocator = mtick.FixedLocator(np.arange(int(round(min(lon))),int(round(max(lon)))+dlon,dlon))
        gl.ylocator = mtick.FixedLocator(np.arange(int(round(min(lat))),int(round(max(lat)))+dlat,dlat))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        ax.coastlines(resolution='50m')
        ax.set_extent([min(lon), max(lon), min(lat), max(lat)])

        titulo = ' '.join(['Z', str(lvl), label_tiempos[t]])
        plt.title(titulo, fontsize=14)
        fmt = ['.jpg']
        for f in fmt:
            fig.savefig(os.path.join(ruta_figs,name_fig+case.dominio+label_tiempos[t]+f), dpi=150, bbox_inches='tight')

def vert_profile(case, var, every, box):
    """
    Calculates the vertical profile of a variable over a region of choice

    - case:     Docpy.WRF object
    - var:      str, name of the variable
    - every:    int, timestep in hours in which the field is required
    - box:      list, box of [lonmin, lonmax, latmin, latmax] dimensions

    - return: numpy.array, of shape (time, lvl) 

    """

    import numpy as np
    
    #setting time
    timelist, delta_t_hs =  case.timedata()
    assert (every >= delta_t_hs), 'every must be bigger than or equal to delta_t_hs. every = {every} and delta_t_hs = {delta_t_hs}'.format(every=every, delta_t_hs=delta_t_hs)
    inter = int(every/delta_t_hs)
    
    # setting box 
    lonmin, lonmax, latmin, latmax = box
    lat, lon = case.get_latlon()

    x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
    x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1
    y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])
    y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1

    if y_idx_ini>y_idx_fin:
        y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini

    # setting vert profile
    var = case.get_variables(var)[::inter,:,y_idx_ini:y_idx_fin,x_idx_ini:x_idx_fin]
    profile = np.mean(np.mean(var, axis=-1), axis=-1)
    return profile


